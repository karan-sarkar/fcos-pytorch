import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from loss import SigmoidFocalLoss
from argument import get_args
from backbone import vovnet57, vovnet27_slim
from dataset import COCODataset, collate_fn, CustomSubset
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
    losses = []
    for images, targets, ids in pbar:
        model.zero_grad()

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        r = torch.range(0, len(targets) - 1).to(device)
        model.eval()
        pred = model(images.tensors, image_sizes=images.sizes, r=r)
        
        
        model.train()
        
        (loss_dict, _) = model(images.tensors, targets=targets, r=r)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()
        
        loss = loss_cls + loss_box + loss_center 
        losses.append(float(loss))

        pred = [p.to('cpu') for p in pred]
        pbar.set_description(str(sum(losses) / len(losses)))
        preds.update({id: p for id, p in zip(ids, pred)})
    
    print('LOSS', sum(losses) / len(losses))
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
    a = flatten(cls_pred)
    cls_p = a.softmax(1)
    hard = (a * 1.5).softmax(1)
    
    mx = torch.argmax(cls_p, 1)
    mask = mx.ge(1).float().view(-1, 1)
   
    return (l1loss(cls_p, hard), mask.mean())

def train(args, epoch, loader, target_loader, model, c_opt, g_opt, device):
    model.train()

    if get_rank() == 0:
        pbar = tqdm(loader, dynamic_ncols=True)

    else:
        pbar = loader
    
    i = 0
    losses = []
    dlosses = []
    for (images, targets, _), (target_images, target_targets, _) in zip(pbar, target_loader):
    
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        target_images = target_images.to(device)
        target_targets = [target.to(device) for target in target_targets]
        if len(targets) != len(target_targets):
            break
        
        c_opt.zero_grad()
        g_opt.zero_grad()
        
        # Train Bottom + Top
        
        r = torch.range(0, len(targets) - 1).to(device)

        (loss_dict, p) = model(images.tensors, targets=targets, r=r)
        (discrep, source_mask) = harden(p, device)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()
        
        loss = loss_cls + loss_box + loss_center + discrep
        
        loss_reduced = reduce_loss_dict(loss_dict)
        cls = loss_reduced['loss_cls'].mean().item()
        box = loss_reduced['loss_box'].mean().item()
        center = loss_reduced['loss_center'].mean().item()
        
        del loss_cls, loss_box, loss_center, loss_dict, loss_reduced, p
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        c_opt.step()
        g_opt.step()
        
        
        
        
        
        c_opt.zero_grad()
        r = torch.range(0, len(targets) - 1).to(device)

        (loss_dict, _) = model(images.tensors, targets=targets, r=r)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()
        
        loss = loss_cls + loss_box + loss_center 
        
        loss_reduced = reduce_loss_dict(loss_dict)
        cls = loss_reduced['loss_cls'].mean().item()
        box = loss_reduced['loss_box'].mean().item()
        center = loss_reduced['loss_center'].mean().item()
        
        del loss_cls, loss_box, loss_center, loss_dict, loss_reduced
        
        (_, p) = model(target_images.tensors, targets=target_targets, r=r)
        dloss, mask = harden(p, device)
        discrep = dloss.mean().item()
        loss -= dloss
        
        del p, dloss
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        c_opt.step()
        
        for _ in range(4):
            g_opt.zero_grad()
            r = torch.range(0, len(targets) - 1).to(device)
            (loss_dict, _) = model(images.tensors, targets=targets, r=r)
            loss_cls = loss_dict['loss_cls'].mean()
            loss_box = loss_dict['loss_box'].mean()
            loss_center = loss_dict['loss_center'].mean()
            
            loss = loss_cls + loss_box + loss_center 
            
            loss_reduced = reduce_loss_dict(loss_dict)
            cls = loss_reduced['loss_cls'].mean().item()
            box = loss_reduced['loss_box'].mean().item()
            center = loss_reduced['loss_center'].mean().item()
            
            del loss_cls, loss_box, loss_center, loss_dict, loss_reduced
            (_, p) = model(target_images.tensors, targets=target_targets, r=r)
            dloss, _ = harden(p, device)
            discrep = dloss.mean().item()
            loss += dloss
            
            del p, dloss
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            g_opt.step()
        
        
        losses.append(cls + box + center)
        dlosses.append(discrep)
        avg = sum(losses) / len(losses)
        davg = sum(dlosses) / len(dlosses)
        mask = float(mask)
        source_mask = float(source_mask)
        
        if i % 100 == 0:
            torch.save((model, c_opt, g_opt), 'slim_fcos_' + str(args.ckpt + epoch + 1) + '.pth')
        
        if get_rank() == 0:
            pbar.set_description(
                (
                    f'epoch: {epoch + 1}; cls: {cls:.4f};'
                    f'box: {box:.4f}; center: {center:.4f}; '
                    f'avg: {avg:.4f}; davg: {davg:.8f}, discrep: {discrep:.4f}'
                    f'mask: {mask:.4f}; source_mask: {source_mask:.4f};'
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
    print(args.n_class)

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
  
    '''
    source_sample = np.random.permutation(len(source))
    target_sample = np.random.permutation(len(target))
    
    source_set = CustomSubset(source, source_sample[:int(0.9 * len(source_sample))])
    target_set = CustomSubset(target, target_sample[:int(0.9 * len(target_sample))])
    source_valid_set = CustomSubset(source, source_sample[int(0.9 * len(source_sample)):])
    target_valid_set = CustomSubset(target, target_sample[int(0.9 * len(target_sample)):])
    '''

    backbone = vovnet27_slim(pretrained=False)
    model = FCOS(args, backbone)
    model = model.to(device)
    model = nn.DataParallel(model)
    
    bottom = [p for n, p in model.named_parameters() if ('pred' not in n)]
    top = [p for n, p in model.named_parameters() if ('pred' in n)]
    print(len(bottom), len(top))
    
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
        (model, co, go) = torch.load('slim_fcos_' + str(args.ckpt) + '.pth')
        if args.rand_class != 'true':
            c_opt = co
            g_opt = go
        '''    
        else:
            for (n, p), (_, q) in zip(m.named_parameters(), model.named_parameters()):
                if 'head' not in n:
                    q = p.to(device)
        '''
        if isinstance(model, nn.DataParallel):
            model = model.module
        model = nn.DataParallel(model)
    else:
        args.ckpt = 0
    for g in c_opt.param_groups:
        g['lr'] = args.lr
    for g in g_opt.param_groups:
        g['lr'] = args.lr2
    
    for epoch in range(args.epoch): 
        valid(args, epoch, target_valid_loader, target_valid_set, model, device)
        valid(args, epoch, source_valid_loader, source_valid_set, model, device)
        train(args, epoch, source_loader, target_loader, model, c_opt, g_opt, device)
        torch.save((model, c_opt, g_opt), 'slim_fcos_' + str(args.ckpt + epoch + 1) + '.pth')
        
        
        
        
       
        


