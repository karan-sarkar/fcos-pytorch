import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
import torch.nn.functional as F
from tqdm import tqdm

from loss import SigmoidFocalLoss
from argument import get_args
from backbone import vovnet57, vovnet27_slim
from dataset import COCODataset, collate_fn
from model import FCOS
from transform import preset_transform
from evaluate import evaluate
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    DistributedSampler,
    all_gather,
)


def accumulate_predictions(predictions):
    all_predictions = all_gather(predictions)

    if get_rank() != 0:
        return

    predictions = {}

    for p in all_predictions:
        predictions.update(p)

    ids = list(sorted(predictions.keys()))

    if len(ids) != ids[-1] + 1:
        print('Evaluation results is not contiguous')

    predictions = [predictions[i] for i in ids]

    return predictions


@torch.no_grad()
def valid(args, epoch, loader, dataset, m, device):
    if args.distributed:
        model = model.module

    torch.cuda.empty_cache()
    
    if isinstance(m, nn.DataParallel):
        model = m.module
    else:
        model = m

    model.eval()

    pbar = tqdm(loader, dynamic_ncols=True)

    preds = {}

    for images, targets, ids in pbar:
        model.zero_grad()

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        r = torch.range(0, len(targets) - 1).to(device)
        
        pred = model(images.tensors, image_sizes=images.sizes, r=r)

        pred = [p.to('cpu') for p in pred]

        preds.update({id: p for id, p in zip(ids, pred)})

    preds = accumulate_predictions(preds)

    if get_rank() != 0:
        return

    evaluate(dataset, preds)
    del preds

def flatten(cls_pred):
    batch = cls_pred[0].shape[0]
    n_class = cls_pred[0].shape[1]

    cls_flat = []
    for i in range(len(cls_pred)):
        cls_flat.append(cls_pred[i].permute(0, 2, 3, 1).reshape(-1, n_class))

    cls_flat = torch.cat(cls_flat, 0)
    return cls_flat

def discrep(cls_pred1, cls_pred2):
    return 3 * nn.L1Loss()(flatten(cls_pred1).sigmoid(), flatten(cls_pred2).sigmoid())

def freeze(model, section, on):
    i = 0
    for n, p in model.named_parameters():
        if section == "bottom" and 'fpn' not in n and 'head' not in n:
            p.requires_grad = on
        if section == "top" and ('fpn' in n or 'head' in n):
            p.requires_grad = on
        i += 1

focal_loss = SigmoidFocalLoss(2.0, 0.25)
l1loss = nn.L1Loss()

def harden(cls_pred, device):
    batch = cls_pred[0].shape[0]
    cls_p = flatten(cls_pred).sigmoid()
    
    
    mx = torch.argmax(cls_p, 1)
    mask = cls_p.max(1)[0].ge(0.05).float()
    mx = F.one_hot(mx, 10)
    return (torch.mean(torch.abs(cls_p -  mx), 1) * mask).mean()
    

def train(args, epoch, loader, target_loader, model, c_opt, g_opt, device):
    model.train()

    if get_rank() == 0:
        pbar = tqdm(loader, dynamic_ncols=True)

    else:
        pbar = loader
    
    i = 0
    losses = []
    for (images, targets, _), (target_images, target_targets, _) in zip(pbar, target_loader):
        
        if len(targets) != len(target_targets):
            break
        
        c_opt.zero_grad()
        g_opt.zero_grad()
        
        # Train Bottom + Top
        
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        
        target_images = target_images.to(device)
        target_targets = [target.to(device) for target in target_targets]
        r = torch.range(0, len(targets) - 1).to(device)

        loss_dict, _ = model(images.tensors, targets=targets, r=r)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()
        
        loss = loss_cls + loss_box + loss_center 
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        c_opt.step()
        g_opt.step()
        del loss_cls, loss_box, loss_center, loss_dict
        
        # Train Top
        c_opt.zero_grad()
        loss_dict, _ = model(images.tensors, targets=targets, r=r)
        loss_dict2, p = model(target_images.tensors, targets=target_targets, r=r)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()
        
        dloss = harden(p, device)
        
        loss_reduced = reduce_loss_dict(loss_dict2)
        loss_cls_target = loss_reduced['loss_cls'].mean().item()
        loss_box_target = loss_reduced['loss_box'].mean().item()
        loss_center_target = loss_reduced['loss_center'].mean().item()
        

        loss = loss_cls + loss_box + loss_center - dloss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        c_opt.step()
        del loss_cls, loss_box, loss_center, loss_dict2, loss_reduced
        
        # Train Bottom
        for j in range(4):
            g_opt.zero_grad()
            #loss_dict, _ = model(images.tensors, targets=targets, r=r)
            #loss_cls = loss_dict['loss_cls'].mean()
            #loss_box = loss_dict['loss_box'].mean()
            #loss_center = loss_dict['loss_center'].mean()
            
            _, p = model(target_images.tensors, targets=target_targets, r=r)
            dloss = harden(p, device)
            #loss = loss_cls + loss_box + loss_center + dloss
            loss = dloss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            g_opt.step()
            #del loss_cls, loss_box, loss_center
        
        loss_reduced = reduce_loss_dict(loss_dict)
        loss_cls = loss_reduced['loss_cls'].mean().item()
        loss_box = loss_reduced['loss_box'].mean().item()
        loss_center = loss_reduced['loss_center'].mean().item()
        discrep_loss = dloss.item()
        losses.append(loss_cls + loss_box + loss_center + discrep_loss)
        avg = sum(losses) / len(losses)
        del loss_dict, loss_reduced
        
        if i % 100 == 0:
            torch.save((model, c_opt, g_opt), 'mini_fcos_' + str(args.ckpt + epoch + 1) + '.pth')
    
        if get_rank() == 0:
            pbar.set_description(
                (
                    f'epoch: {epoch + 1}; cls: {loss_cls:.4f}; target_cls: {loss_cls_target:.4f};'
                    f'box: {loss_box:.4f}; target_box: {loss_box_target:.4f}; center: {loss_center:.4f}; target_center: {loss_center_target:.4f};'
                    f'discrepancy: {discrep_loss:.4f}'
                    f'avg: {avg:.4f}'
                )
            )
        i+= 1


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return sampler.RandomSampler(dataset)

    else:
        return sampler.SequentialSampler(dataset)

if __name__ == '__main__':
    args = get_args()

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = False

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_set = COCODataset(args.path, 'train', preset_transform(args, train=True))
    target_set = COCODataset(args.path2, 'train', preset_transform(args, train=True))
    source_valid_set = COCODataset(args.path, 'val', preset_transform(args, train=False))
    target_valid_set = COCODataset(args.path2, 'val', preset_transform(args, train=False))

    backbone = vovnet27_slim(pretrained=True)
    model = FCOS(args, backbone)
    model = nn.DataParallel(model)
    
    bottom = [p for n, p in model.named_parameters() if ('fpn' not in n and 'head' not in n)]
    top = [p for n, p in model.named_parameters() if ('fpn' in n or 'head' in n)]

    g_opt = optim.SGD(
        bottom,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.l2,
        nesterov=True,
    )
    
    c_opt = optim.SGD(
        top,
        lr=args.lr2,
        momentum=0.9,
        weight_decay=args.l22,
        nesterov=True,
    )
    

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
    

    source_loader = DataLoader(
        source_set,
        batch_size=args.batch,
        sampler = data_sampler(source_set, True, args.distributed),
        collate_fn=collate_fn(args),
    )
    target_loader = DataLoader(
        target_set,
        batch_size=args.batch,
        sampler = data_sampler(target_set, True, args.distributed),
        collate_fn=collate_fn(args),
    )
    source_valid_loader = DataLoader(
        source_valid_set,
        batch_size=args.batch_val,
        shuffle = True,
        collate_fn=collate_fn(args),
    )
    target_valid_loader = DataLoader(
        target_valid_set,
        batch_size=args.batch_val,
        shuffle = True,
        collate_fn=collate_fn(args),
    )
    
    if args.ckpt is not None:
        (model, c_opt, g_opt) = torch.load('mini_fcos_' + str(args.ckpt) + '.pth')
        #if isinstance(model, nn.DataParallel):
            #model = model.module
        if not isinstance(model, nn.DataParallel):
            model = nn.DataParallel(model)
    else:
        args.ckpt = 0
    for g in c_opt.param_groups:
        g['lr'] = args.lr
    for g in g_opt.param_groups:
        g['lr'] = args.lr2
    model = model.to(device)
    
    for epoch in range(args.epoch):
        valid(args, epoch, target_valid_loader, target_valid_set, model, device)
        valid(args, epoch, source_valid_loader, source_valid_set, model, device)
        train(args, epoch, source_loader, target_loader, model, c_opt, g_opt, device)
        torch.save((model, c_opt, g_opt), 'mini_fcos_' + str(args.ckpt + epoch + 1) + '.pth')
        
       
        


