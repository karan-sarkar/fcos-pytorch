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


def train(args, epoch, loader, unlabeled_loader, model, opt, device):
    model.train()

    if get_rank() == 0:
        pbar = tqdm(zip(loader, unlabeled_loader), dynamic_ncols=True)

    else:
        pbar = loader
    
    i = 0
    losses = []
    dlosses = []
    for ((images, aug_images, targets), (unlabeled_images, unlabeled_aug_images, _)) in pbar:
        images = images.to(device)
        aug_images = aug_images.to(device)
        unlabeled_images = unlabeled_images.to(device)
        unlabeled_aug_images = unlabeled_aug_images.to(device)
        targets = [target.to(device) for target in targets]
        
        opt.zero_grad()
        
        # Train Bottom + Top
        
        r = torch.range(0, len(targets) - 1).to(device)

        (loss_dict, p) = model(images.tensors, targets=targets, r=r)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()
        
        loss = loss_cls + loss_box + loss_center 
        
        loss_reduced = reduce_loss_dict(loss_dict)
        cls = loss_reduced['loss_cls'].mean().item()
        box = loss_reduced['loss_box'].mean().item()
        center = loss_reduced['loss_center'].mean().item()
        
        del loss_cls, loss_box, loss_center, loss_dict, loss_reduced, p
        
        (loss_dict, p) = model(aug_images.tensors, targets=targets, r=r)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()
        
        loss += loss_cls + loss_box + loss_center 
        del loss_cls, loss_box, loss_center, loss_dict, p
        del images, aug_images
        
        
        with torch.no_grad():
            model.eval()
            preds = model(unlabeled_images.tensors, image_sizes=unlabeled_images.sizes, r=r)
        model.train()
        
        if len(preds) > 0:
            (loss_dict, p) = model(unlabeled_aug_images.tensors, targets=preds, r=r)
            loss_cls = loss_dict['loss_cls'].mean()
            loss_box = loss_dict['loss_box'].mean()
            loss_center = loss_dict['loss_center'].mean()
            
            discrep = loss_cls + loss_box + loss_center 
        else:
            discrep = 0
        loss += discrep
       
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        opt.step()
        del loss_cls, loss_box, loss_center, loss_dict, p
        
        
        losses.append(cls + box + center)
        dlosses.append(discrep)
        avg = sum(losses) / len(losses)
        davg = sum(dlosses) / len(dlosses)
        
        if i % 100 == 0:
            torch.save((model, opt), 'fix_fcos_' + str(args.ckpt + epoch + 1) + '.pth')
        
        if get_rank() == 0:
            pbar.set_description(
                (
                    f'epoch: {epoch + 1}; cls: {cls:.4f};'
                    f'box: {box:.4f}; center: {center:.4f}; '
                    f'avg: {avg:.4f}; davg: {davg:.8f}, discrep: {discrep:.4f}'
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
    source_aug_set = COCODataset(args.path, 'train', preset_transform(args, train=True, augment = True))
    source_valid_set = COCODataset(args.path, 'val', preset_transform(args, train=False))
  
    sample = np.random.choice(len(source_set), 1000, replace=False)
    source_train_set = AugmentedDataset(source_set, source_aug_set, sample)
    unlabeled_set = AugmentedDataset(source_set, source_aug_set, np.arange(len(source_set)))
    
    backbone = vovnet27_slim(pretrained=False)
    model = FCOS(args, backbone)
    model = model.to(device)
    #model = nn.DataParallel(model)
    
    opt = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.l2,
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
        source_train_set,
        batch_size=args.batch,
        sampler = data_sampler(source_train_set, True, args.distributed),
        collate_fn=collate_fx(args),
    )
    unlabeled_loader = DataLoader(
        unlabeled_set,
        batch_size=args.batch,
        sampler = data_sampler(unlabeled_set, True, args.distributed),
        collate_fn=collate_fx(args),
    )
    source_valid_loader = DataLoader(
        source_valid_set,
        batch_size=args.batch_val,
        shuffle = True,
        collate_fn=collate_fn(args),
    )
    
    if args.ckpt is not None:
        (model, o) = torch.load('fix_fcos_' + str(args.ckpt) + '.pth')
        if args.rand_class != 'true':
            opt = o
        if isinstance(model, nn.DataParallel):
            model = model.module
        model = nn.DataParallel(model)
    else:
        args.ckpt = 0
    for g in opt.param_groups:
        g['lr'] = args.lr
    
    for epoch in range(args.epoch):
        train(args, epoch, source_loader, unlabeled_loader, model, opt, device)
        torch.save((model, opt), 'fix_fcos_' + str(args.ckpt + epoch + 1) + '.pth')
        valid(args, epoch, source_valid_loader, source_valid_set, model, device)
        
        
       
        


