import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from AugmentedDataset import AugmentedDataset

from loss import SigmoidFocalLoss
from argument import get_args
from backbone import vovnet57, vovnet27_slim
from dataset import COCODataset, collate_fx, collate_fn, CustomSubset
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

from ema import ModelEMA



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

        preds.update({id: p for id, p in zip(ids, pred)})
    
    print('LOSS', sum(losses) / len(losses))
    preds = accumulate_predictions(preds)

    if get_rank() != 0:
        return

    evaluate(dataset, preds)
    del preds


def freeze(model, section, on):
    i = 0
    for n, p in model.named_parameters():
        if section == "bottom" and 'fpn' not in n and 'head' not in n:
            p.requires_grad = on
        if section == "top" and ('fpn' in n or 'head' in n):
            p.requires_grad = on
        i += 1

focal_loss = SigmoidFocalLoss(2.0, 0.25)
l1loss = nn.L1Loss(reduction='none')
bceloss = nn.BCELoss(reduction='none')

def flatten(cls_pred):
    batch = cls_pred[0].shape[0]
    n_class = cls_pred[0].shape[1]

    cls_flat = []
    for i in range(len(cls_pred)):
        cls_flat.append(cls_pred[i].permute(0, 2, 3, 1).reshape(-1, n_class))

    cls_flat = torch.cat(cls_flat, 0)
    return cls_flat

def compare(cls_pred1, cls_pred2):
    batch = cls_pred1[0].shape[0]
    cls_p1 = flatten(cls_pred1).sigmoid()
    cls_p2 = flatten(cls_pred2).sigmoid()
    if cls_p1.shape[0] != cls_p2.shape[0]:
        return (0, 0)
    
    mx = torch.argmax(cls_p1, 1)
    mask = cls_p1.max(1)[0].ge(0.25).float()
    mx = F.one_hot(mx, 10)
    return ((bceloss(cls_p2, mx) * mask).mean(), float(mask.mean()))

def train(args, epoch, loader, target_loader, model, ema_model, c_opt, g_opt, device):
    model.train()

    pbar = tqdm(loader, dynamic_ncols=True)
    
    i = 0
    losses = []
    dlosses = []
    for ((images, targets, _), (target_images, target_aug_images, target_targets)) in zip(pbar, target_loader):
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        target_images = target_images.to(device)
        target_aug_images = target_aug_images.to(device)
        target_targets = [target.to(device) for target in target_targets]
        if len(targets) != args.batch or len(target_targets) != args.batch_val:
            break
        
        c_opt.zero_grad()
        g_opt.zero_grad()
        
        # Train Bottom + Top
        
        r = torch.range(0, len(targets) - 1).to(device)

        (loss_dict, p) = model(images.tensors, targets=targets, r=r)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()
        
        loss = loss_cls + loss_box + loss_center 
        
        loss_reduced = reduce_loss_dict(loss_dict)
        cls = float(loss_reduced['loss_cls'].mean().item())
        box = float(loss_reduced['loss_box'].mean().item())
        center = float(loss_reduced['loss_center'].mean().item())
        
        del loss_cls, loss_box, loss_center, loss_dict, loss_reduced, p
        
        
        with torch.no_grad():
            (_, p) = model(target_images.tensors, targets=target_targets, r=r)
            preds = [pred.to(device) for pred in preds]
        (_, q) = model(target_aug_images.tensors, targets=target_targets, r=r)
        discrep, mask = compare(p, q)
        del p, q
        
        
        loss.backward()
        
        if sum([p.grad.isnan().any() for p in model.parameters()]) == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            c_opt.step()
            g_opt.step()
            ema_model.update(model)
        
        if loss.isnan().sum() + discrep.isnan().sum() > 0:
            (model, c_opt, g_opt, ema_model) = torch.load('style_fcos_' + str(args.ckpt + epoch + 1) + '.pth')
            
        del discrep
            
        g_opt.zero_grad()
        (loss_dict, p, source_style) = model(images.tensors, targets=targets, r=r, style=True)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()
        
        loss = loss_cls + loss_box + loss_center 
        
        del loss_cls, loss_box, loss_center, loss_dict, p
        del images, targets
        
        with torch.no_grad():
            (_, p, target_style) = model(target_images.tensors, targets=target_targets, r=r, style=True)
            preds = [pred.to(device) for pred in preds]
        (_, q) = model(target_aug_images.tensors, targets=target_targets, r=r)
        discrep, mask = compare(p, q)
        del p, q
        
        
        #0.00001
        style_loss = args.mul * (source_style.mean(0) - target_style.mean(0)).pow(2).mean()
        loss +=  style_loss
        del source_style, target_style
        loss.backward()
        
        if sum([p.grad.isnan().any() for p in model.parameters()]) == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            g_opt.step()
            ema_model.update(model)
        
        if loss.isnan().sum() + discrep.isnan().sum() > 0:
            (model, c_opt, g_opt, ema_model) = torch.load('style_fcos_' + str(args.ckpt + epoch + 1) + '.pth')
            
            
            
            
            
            
        
        losses.append(cls + box + center)
        dlosses.append(float(discrep))
        discrep = float(discrep)
        style = float(style_loss)
        avg = sum(losses) / len(losses)
        davg = sum(dlosses) / len(dlosses)
        
        if i % 100 == 0:
            torch.save((model, c_opt, g_opt, ema_model), 'style_fcos_' + str(args.ckpt + epoch + 1) + '.pth')
        
        if get_rank() == 0:
            pbar.set_description(
                (
                    f'epoch: {epoch + 1}; cls: {cls:.4f};'
                    f'box: {box:.4f}; center: {center:.4f}; '
                    f'avg: {avg:.4f}; davg: {davg:.8f}, discrep: {discrep:.4f}; style: {style:.4f}'
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
    source_valid_set = COCODataset(args.path, 'val', preset_transform(args, train=False))
    
    target_set = COCODataset(args.path2, 'train', preset_transform(args, train=True))
    target_aug_set = COCODataset(args.path2, 'train', preset_transform(args, train=True, augment = True))
    target_valid_set = COCODataset(args.path2, 'val', preset_transform(args, train=False))
  
    sample = np.random.choice(len(target_set), len(target_set), replace=False)
    target_set = AugmentedDataset(target_set, target_aug_set, sample)
    
    backbone = vovnet27_slim(pretrained=False)
    a = torch.load('slim_fcos_66.pth')
    model = a[0]
    if isinstance(model, nn.DataParallel):
        model = model.module
    model = nn.DataParallel(model)
    ema_model = ModelEMA(model, 0.999, device)
    
    bottom = [p for n, p in model.named_parameters() if ('head' not in n)]
    top = [p for n, p in model.named_parameters() if ('head' in n)]
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
        batch_size=args.batch_val,
        sampler = data_sampler(target_set, True, args.distributed),
        collate_fn=collate_fx(args),
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
        (model, c, g, ema_model) = torch.load('style_fcos_' + str(args.ckpt) + '.pth')
        if args.rand_class != 'true':
            c_opt = c
            g_opt = g
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
        #valid(args, epoch, target_valid_loader, target_valid_set, ema_model.ema, device)
        #valid(args, epoch, source_valid_loader, source_valid_set, ema_model.ema, device)
        train(args, epoch, source_loader, target_loader, model, ema_model, c_opt, g_opt, device)
        torch.save((model, c_opt, g_opt, ema_model), 'style_fcos_' + str(args.ckpt + epoch + 1) + '.pth')
        
        
       
        


