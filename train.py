import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm

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
def valid(args, epoch, loader, dataset, model, device):
    if args.distributed:
        model = model.module

    torch.cuda.empty_cache()

    model.eval()

    pbar = tqdm(loader, dynamic_ncols=True)

    preds = {}

    for images, targets, ids in pbar:
        model.zero_grad()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        pred, _ = model(images.tensors, images.sizes)

        pred = [p.to('cpu') for p in pred]

        preds.update({id: p for id, p in zip(ids, pred)})

    preds = accumulate_predictions(preds)

    if get_rank() != 0:
        return

    evaluate(dataset, preds)

def flatten(cls_pred):
    batch = cls_pred[0].shape[0]
    n_class = cls_pred[0].shape[1]

    cls_flat = []
    for i in range(len(cls_pred)):
        cls_flat.append(cls_pred[i].permute(0, 2, 3, 1).reshape(-1, n_class))

    cls_flat = torch.cat(cls_flat, 0)
    return cls_flat
    
def compare(out, t):
    n_class = out.shape[1]
    class_ids = torch.arange(
        1, n_class + 1, dtype=t.dtype, device=t.device
    ).unsqueeze(0)
    
    p = torch.sigmoid(out)

    gamma = self.gamma
    alpha = self.alpha

    term1 = (1 - p) ** gamma * torch.log(p)
    term2 = p ** gamma * torch.log(1 - p)

    # print(term1.sum(), term2.sum())

    loss = (
        -(t == class_ids).float() * alpha * term1
        - ((t != class_ids) * (t >= 0)).float() * (1 - alpha) * term2
    )

    return loss.sum()

def discrep(cls_pred1, cls_pred2):
    cls_flat1 = flatten(cls_pred1)
    cls_flat2 = flatten(cls_pred2)
    return compare(cls_flat1, cls_flat2.argmax(-1)) + compare(cls_flat2, cls_flat1.argmax(-1))

def train(args, epoch, loader, target_loader, model, optimizer, device):
    model.train()

    if get_rank() == 0:
        pbar = tqdm(loader, dynamic_ncols=True)

    else:
        pbar = loader
    
    for (images, targets, _), (target_images, target_targets, _) in zip(pbar, target_loader):
        model.zero_grad()
        
        # Train Bottom + Top
        
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        
        target_images = target_images.to(device)
        target_targets = [target.to(device) for target in target_targets]

        loss_dict2, loss_dict, _, _ = model(images.tensors, targets=targets)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()
        
        loss_cls2 = loss_dict2['loss_cls'].mean()
        loss_box2 = loss_dict2['loss_box'].mean()
        loss_center2 = loss_dict2['loss_center'].mean()

        loss = loss_cls + loss_box + loss_center + loss_cls2 + loss_box2 + loss_center2
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        
        # Train Top
        model.freeze("bottom", False)
        loss_dict2, loss_dict, _, _ = model(images.tensors, targets=targets)
        _, _, p1, p2 = model(target_images.tensors, targets=target_targets)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()
        
        loss_cls2 = loss_dict2['loss_cls'].mean()
        loss_box2 = loss_dict2['loss_box'].mean()
        loss_center2 = loss_dict2['loss_center'].mean()
        dloss = discrep(p1, p2)

        loss = loss_cls + loss_box + loss_center + loss_cls2 + loss_box2 + loss_center2 - dloss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
        model.freeze("bottom", True)
        
        # Train Bottom
        model.freeze("top", False)
        for _ in range(4):
            loss_dict2, loss_dict, _, _ = model(images.tensors, targets=targets)
            _, _, p1, p2 = model(target_images.tensors, targets=target_targets)
            loss_cls = loss_dict['loss_cls'].mean()
            loss_box = loss_dict['loss_box'].mean()
            loss_center = loss_dict['loss_center'].mean()
            
            loss_cls2 = loss_dict2['loss_cls'].mean()
            loss_box2 = loss_dict2['loss_box'].mean()
            loss_center2 = loss_dict2['loss_center'].mean()
            dloss = discrep(p1, p2)

            loss = loss_cls + loss_box + loss_center + loss_cls2 + loss_box2 + loss_center2 + dloss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
        model.freeze("top", True)
        
        loss_reduced = reduce_loss_dict(loss_dict)
        loss_cls = loss_reduced['loss_cls'].mean().item()
        loss_box = loss_reduced['loss_box'].mean().item()
        loss_center = loss_reduced['loss_center'].mean().item()
        discrep_loss = dloss.item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'epoch: {epoch + 1}; cls: {loss_cls:.4f}; '
                    f'box: {loss_box:.4f}; center: {loss_center:.4f}'
                    f'discrepancy: {discrep_loss:.4f}'
                )
            )


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
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    device = 'cuda'

    source_set = COCODataset(args.path, 'train', preset_transform(args, train=True))
    target_set = COCODataset(args.path2, 'train', preset_transform(args, train=True))
    source_valid_set = COCODataset(args.path2, 'val', preset_transform(args, train=False))
    target_valid_set = COCODataset(args.path2, 'val', preset_transform(args, train=False))

    backbone = vovnet57(pretrained=False)
    model = FCOS(args, backbone)
    model = model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.l2,
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
        sampler=data_sampler(source_set, shuffle=True, distributed=args.distributed),
        num_workers=2,
        collate_fn=collate_fn(args),
    )
    target_loader = DataLoader(
        target_set,
        batch_size=args.batch,
        sampler=data_sampler(target_set, shuffle=False, distributed=args.distributed),
        num_workers=2,
        collate_fn=collate_fn(args),
    )
    source_valid_loader = DataLoader(
        source_valid_set,
        batch_size=args.batch,
        sampler=data_sampler(source_valid_set, shuffle=True, distributed=args.distributed),
        num_workers=2,
        collate_fn=collate_fn(args),
    )
    target_valid_loader = DataLoader(
        target_valid_set,
        batch_size=args.batch,
        sampler=data_sampler(target_valid_set, shuffle=False, distributed=args.distributed),
        num_workers=2,
        collate_fn=collate_fn(args),
    )
    
    
    for epoch in range(args.epoch):
        train(args, epoch, source_loader, target_loader, model, optimizer, device)
        valid(args, epoch, source_valid_loader, source_valid_set, model, device)
        valid(args, epoch, target_valid_loader, target_valid_set, model, device)

        scheduler.step()

        if get_rank() == 0:
            torch.save(
                {'model': model.module.state_dict(), 'optim': optimizer.state_dict()},
                f'checkpoint/epoch-{epoch + 1}.pt',
            )

