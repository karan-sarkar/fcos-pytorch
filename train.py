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
from shallow import Shallow
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


l1loss = nn.L1Loss()
bceloss = nn.BCELoss()

@torch.no_grad()
def valid(args, epoch, loader, dataset, m, device):
    if get_rank() == 0:
        pbar = tqdm(loader, dynamic_ncols=True)

    else:
        pbar = loader
    
    i = 0
    losses = []
    dlosses = []
    for (images, targets, _) in pbar:
        with torch.no_grad():
            images = images.to(device)
            targets = targets.to(device)
            
            mask = model(images).sigmoid(1).max(1)[0].ge(0.5).float()
            source_loss = l1loss(mask, targets)
            
            losses.append(float(source_loss))
            avg = sum(losses) / len(losses)
            
        
            if get_rank() == 0:
                pbar.set_description(
                    (
                        f'avg: {avg:.4f};'
                    )
                )
            i+= 1



def discrep(x):
    mask = x.sigmoid(1).max(1)[0].ge(0.5).float()
    return l1loss(x.sigmoid(1), mask)

def train(args, epoch, loader, target_loader, model, c_opt, g_opt, device):
    if get_rank() == 0:
        pbar = tqdm(loader, dynamic_ncols=True)

    else:
        pbar = loader
    
    i = 0
    losses = []
    dlosses = []
    for (images, targets, _), (target_images, target_targets, _) in zip(pbar, target_loader):
        
        # Train Bottom + Top
        
        images = images.to(device)
        targets = targets.to(device)
        target_images = target_images.to(device)
        target_targets = target_targets.to(device)
        
        c_opt.zero_grad()
        g_opt.zero_grad()
        loss = bceloss(model(images).view(-1).sigmoid(), targets.view(-1))
        loss.backward()
        c_opt.step()
        g_opt.step()
        
        c_opt.zero_grad()
        source_loss = bceloss(model(images).view(-1).sigmoid(), targets.view(-1))
        target_loss = discrep(model(target_images))
        loss = source_loss - target_loss
        loss.backward()
        c_opt.step()
        
        for _ in range(4):
            g_opt.zero_grad()
            target_loss = discrep(model(target_images))
            loss = target_loss
            loss.backward()
            g_opt.step()
        
        
        
        losses.append(float(source_loss))
        avg = sum(losses) / len(losses)
        dlosses.append(float(target_loss))
        davg = sum(dlosses) / len(dlosses)
        
        if i % 100 == 0:
            torch.save((model, c_opt, g_opt), 'shallow_' + str(args.ckpt + epoch + 1) + '.pth')
    
        if get_rank() == 0:
            pbar.set_description(
                (
                    f'avg: {avg:.4f}; davg: {davg:.4f};'
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

def filter(n):
    return ('layer4' in n or 'fc' in n)

if __name__ == '__main__':
    args = get_args()

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = False

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    '''
    source_set = COCODataset(args.path, 'train', preset_transform(args, train=True))
    target_set = COCODataset(args.path2, 'train', preset_transform(args, train=True))
    source_valid_set = COCODataset(args.path, 'val', preset_transform(args, train=True))
    target_valid_set = COCODataset(args.path2, 'val', preset_transform(args, train=True))
  
    '''
    
    source = COCODataset(args.path, 'train', preset_transform(args, train=True))
    target = COCODataset(args.path2, 'train', preset_transform(args, train=True))
    
    print(len(source), len(target))
    
    source_sample = np.random.permutation(len(source))
    target_sample = np.random.permutation(len(target))
    
    print(source_sample.shape)
    
    source_set = torch.utils.data.Subset(source, source_sample[:int(0.9 * len(source_sample))])
    target_set = torch.utils.data.Subset(target, target_sample[:int(0.9 * len(target_sample))])
    source_valid_set = torch.utils.data.Subset(source, source_sample[int(0.9 * len(source_sample)):])
    target_valid_set = torch.utils.data.Subset(target, target_sample[int(0.9 * len(target_sample)):])
    

    backbone = vovnet27_slim(pretrained=False)
    model = Shallow(args)
    model = model.to(device)
    model = nn.DataParallel(model)
    
    bottom = [p for n, p in model.named_parameters() if not filter(n)]
    top = [p for n, p in model.named_parameters() if filter(n)]

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
        (model, c_opt, g_opt) = torch.load('shallow_' + str(args.ckpt) + '.pth')
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
        train(args, epoch, source_loader, target_loader, model, c_opt, g_opt, device)
        torch.save((model, c_opt, g_opt), 'shallow_' + str(args.ckpt + epoch + 1) + '.pth')
        valid(args, epoch, source_valid_loader, source_valid_set, model, device)
        valid(args, epoch, target_valid_loader, target_valid_set, model, device)
        
       
        


