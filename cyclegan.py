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
from unet import Unet
import torchvision.models as models

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
def valid(args, epoch, loader, dataset, m, G1, device):
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
        images = G1(images)
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

bceloss = nn.BCEWithLogitsLoss()

def high(x):
    return bceloss(x, torch.ones_like(x))

def low(x):
    return bceloss(x, torch.zeros_like(x))

def train(args, epoch, loader, target_loader, cyclegan, c_opt, g_opt, device):
    C1, C2, G1, G2 = cyclegan

    if get_rank() == 0:
        pbar = tqdm(loader, dynamic_ncols=True)

    else:
        pbar = loader
    
    i = 0
    losses = []
    dlosses = []
    for (source_x, _, _), (target_x, _, _) in zip(pbar, target_loader):
        
        if len(targets) != len(target_targets):
            break
        
        source_x = source_x.to(device)
        target_x = target_x.to(device)
        
        # Train Discriminators
        c_opt.zero_grad()
        loss = high(C1(source_x)) + high(C2(target_x)) + low(C1(G1(target_x))) + low(C2(G2(source_x)))
        loss.backward()
        c_opt.step()
        
        # Train Generators
        g_opt.zero_grad()
        fake_source_x = G1(target_x)
        fake_target_x = G2(source_x)
        gen_loss = high(C1(fake_source_x)) + high(C2(fake_target_x))
        rec_loss = (source_x - G1(fake_target_x)).square().mean() + (target_x - G2(fake_source_x)).square().mean()
        loss = gen_loss + rec_loss
        loss.backward()
        g_opt.step()
        
        
      
        if i % 100 == 0:
            torch.save((C1, C2, G1, G2, c_opt, g_opt), 'cyclegan_' + str(args.ckpt + epoch + 1) + '.pth')
    
        if get_rank() == 0:
            pbar.set_description(
                (
                    f'gen_loss: {gen_loss.item():.4f}; rec_loss: {rec_loss.item():.4f};'
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
    source_valid_set = COCODataset(args.path, 'val', preset_transform(args, train=True))
    target_valid_set = COCODataset(args.path2, 'val', preset_transform(args, train=True))
  
    '''
    source_sample = np.random.permutation(len(source))
    target_sample = np.random.permutation(len(target))
    
    source_set = CustomSubset(source, source_sample[:int(0.9 * len(source_sample))])
    target_set = CustomSubset(target, target_sample[:int(0.9 * len(target_sample))])
    source_valid_set = CustomSubset(source, source_sample[int(0.9 * len(source_sample)):])
    target_valid_set = CustomSubset(target, target_sample[int(0.9 * len(target_sample)):])
    '''

    G1, G2 = Unet(in_chans=3, out_chans=3, chans=args.chan), Unet(in_chans=3, out_chans=3, chans=args.chan)
    
    C1 = models.resnet18(pretrained=True)
    C1.fc = nn.Linear(C1.fc.data.shape[0], 1)
    C2 = models.resnet18(pretrained=True)
    C2.fc = nn.Linear(C2.fc.data.shape[0], 1)
    
    C1, C2, G1, G2 = nn.DataParallel(C1.to(device)), nn.DataParallel(C2.to(device)), nn.DataParallel(G1.to(device)), nn.DataParallel(G2.to(device))
    
    g_opt = optim.SGD(
        list(G1.parameters()) + list(G2.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.l2,
        nesterov=True,
    )
    
    c_opt = optim.SGD(
        list(C1.parameters()) + list(C2.parameters()),
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
    
    (model, _, _) = torch.load('baseline_models/slim_fcos_67.pth')
    model = model.to(device)
    
    if args.ckpt is not None:
        (C1, C2, G1, G2, c_opt, g_opt) = torch.load('cyclegan_' + str(args.ckpt) + '.pth')
        if isinstance(C1, nn.DataParallel):
            C1, C2, G1, G2 = C1.module, C2.module, G1.module, G2.module
        C1, C2, G1, G2 = nn.DataParallel(C1.to(device)), nn.DataParallel(C2.to(device)), nn.DataParallel(G1.to(device)), nn.DataParallel(G2.to(device))
    
    else:
        args.ckpt = 0
    for g in c_opt.param_groups:
        g['lr'] = args.lr
    for g in g_opt.param_groups:
        g['lr'] = args.lr2
    
    for epoch in range(args.epoch):
        train(args, epoch, source_loader, target_loader, (C1, C2, G1, G2), c_opt, g_opt, device)
        torch.save((C1, C2, G1, G2, c_opt, g_opt), 'cyclegan_' + str(args.ckpt + epoch + 1) + '.pth')
        valid(args, epoch, target_valid_loader, target_valid_set, model, G1, device)
        
       
        


