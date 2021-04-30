import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm

from loss import SigmoidFocalLoss
from argument import get_args
from backbone import vovnet57
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

        pred = model(images.tensors, image_sizes=images.sizes)

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

def harden(cls_pred, device):
    batch = cls_pred[0].shape[0]
    loss = 0
    hits = 0
    for cls_flat in cls_pred:
        for i in range(batch):
            cls_p = cls_flat[i]
            cls_p = cls_p.view(10, -1).transpose(0, 1)
            maxs = cls_p.sigmoid().max(-1)[0]
            top_n = (maxs > 0.05).sum().clamp(max = 100)
            idx = maxs.topk(top_n)[1]
            clusters = torch.zeros(cls_p.size(0)).int().to(device)
            clusters[idx] = ((cls_p.argmax(-1) + 1)[idx]).int()
            pos_id = torch.nonzero(clusters > 0).squeeze(1)
            hits += clusters.numel()
            loss += focal_loss(cls_p, clusters) 
    return 10 * loss / (hits)
    

def train(args, epoch, loader, target_loader, model, optimizer, optimizer2, device):
    model.train()

    if get_rank() == 0:
        pbar = tqdm(loader, dynamic_ncols=True)

    else:
        pbar = loader
    
    i = 0
    losses = []
    for (images, targets, _), (target_images, target_targets, _) in zip(pbar, target_loader):
        
        optimizer.zero_grad()
        
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
        optimizer.step()
        del loss_cls, loss_box, loss_center, loss_dict
        
        # Train Top
        freeze(model, "bottom", False)
        optimizer.zero_grad()
        loss_dict, _ = model(images.tensors, targets=targets, r=r)
        _, p = model(target_images.tensors, targets=target_targets, r=r)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()
        
        dloss = harden(p, device)

        loss = loss_cls + loss_box + loss_center - dloss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        freeze(model, "bottom", True)
        del loss_cls, loss_box, loss_center, loss_dict, 
        
        # Train Bottom
        freeze(model, "top", False)
        for j in range(2):
            optimizer2.zero_grad()
            loss_dict, _ = model(images.tensors, targets=targets, r=r)
            loss_cls = loss_dict['loss_cls'].mean()
            loss_box = loss_dict['loss_box'].mean()
            loss_center = loss_dict['loss_center'].mean()
            
            _, p = model(target_images.tensors, targets=target_targets, r=r)
            dloss = harden(p, device)
            loss = loss_cls + loss_box + loss_center + dloss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer2.step()
            del loss_cls, loss_box, loss_center
        freeze(model, "top", True)
        
        loss_reduced = reduce_loss_dict(loss_dict)
        loss_cls = loss_reduced['loss_cls'].mean().item()
        loss_box = loss_reduced['loss_box'].mean().item()
        loss_center = loss_reduced['loss_center'].mean().item()
        discrep_loss = dloss.item()
        losses.append(loss_cls + loss_box + loss_center + discrep_loss)
        avg = sum(losses) / len(losses)
        del loss_dict, loss_reduced
        
        if i % 100 == 0:
            torch.save((model, optimizer, optimizer2), 'fcos_' + str(args.ckpt + epoch + 1) + '.pth')

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'epoch: {epoch + 1}; cls: {loss_cls:.4f}; '
                    f'box: {loss_box:.4f}; center: {loss_center:.4f}'
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

    backbone = vovnet57(pretrained=False)
    model = FCOS(args, backbone)
    

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.l2,
        nesterov=True,
    )
    
    optimizer2 = optim.SGD(
        model.parameters(),
        lr=args.lr2,
        momentum=0.9,
        weight_decay=args.l22,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[16, 22], gamma=0.1
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
        batch_size=args.batch,
        shuffle = True,
        collate_fn=collate_fn(args),
    )
    target_valid_loader = DataLoader(
        target_valid_set,
        batch_size=args.batch,
        shuffle = True,
        collate_fn=collate_fn(args),
    )
    
    if args.ckpt is not None:
        (model, optimizer, optimizer2) = torch.load('fcos_' + str(args.ckpt) + '.pth')
        if isinstance(model, nn.DataParallel):
            model = model.module
        #if not isinstance(model, nn.DataParallel):
            #model = nn.DataParallel(model)
        model = model.to(device)
    else:
        args.ckpt = 0
    for g in optimizer.param_groups:
        g['lr'] = args.lr
    for g in optimizer2.param_groups:
        g['lr'] = args.lr2
    
    for epoch in range(args.epoch):
        train(args, epoch, source_loader, target_loader, model, optimizer, optimizer2, device)
        torch.save((model, optimizer, optimizer2), 'fcos_' + str(args.ckpt + epoch + 1) + '.pth')
        valid(args, epoch, source_valid_loader, source_valid_set, model, device)
        valid(args, epoch, target_valid_loader, target_valid_set, model, device)
        scheduler.step()


