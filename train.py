import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from loss import SigmoidFocalLoss
from argument import get_args
from backbone import vovnet57, vovnet27_slim, vovnet39
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
        
        (loss_dict, _), (loss_dict2, _) = model(images.tensors, targets=targets, r=r)
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

def flatten(cls_pred, col):
    batch = cls_pred[0].shape[0]
    n_class = cls_pred[0].shape[1]

    cls_flat = []
    for i in range(len(cls_pred)):
        cls_flat.append(cls_pred[i].permute(0, 2, 3, 1).reshape(-1, col))

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

def rel_l1(x, y):
    diff = (x - y) / (x + y)
    return diff.abs().mean()

def process(location, cls_pred, box_pred, center_pred):
    batch, channel, height, width = cls_pred.shape

    cls_pred = cls_pred.view(batch, channel, height, width).permute(0, 2, 3, 1)
    cls_pred = cls_pred.reshape(batch, -1, channel).softmax(-1).contiguous()[:, :, 1:]

    box_pred = box_pred.view(batch, 4, height, width).permute(0, 2, 3, 1)
    box_pred = box_pred.reshape(batch, -1, 4)

    center_pred = center_pred.view(batch, 1, height, width).permute(0, 2, 3, 1)
    center_pred = center_pred.reshape(batch, -1).sigmoid()

    candid_ids = cls_pred > 0.05
    top_ns = candid_ids.view(batch, -1).sum(1)
    top_ns = top_ns.clamp(max=1000)

    cls_pred = cls_pred * center_pred[:, :, None]

    results = []

    for i in range(batch):
        cls_p = cls_pred[i]
        candid_id = candid_ids[i]
        cls_p = cls_p[candid_id]
        candid_nonzero = candid_id.nonzero()
        box_loc = candid_nonzero[:, 0]
        class_id = candid_nonzero[:, 1] + 1

        box_p = box_pred[i]
        box_p = box_p[box_loc]
        loc = location[box_loc]

        top_n = top_ns[i]

        if candid_id.sum().item() > top_n.item():
            cls_p, top_k_id = cls_p.topk(top_n, sorted=False)
            class_id = class_id[top_k_id]
            box_p = box_p[top_k_id]
            loc = loc[top_k_id]

        detections = torch.stack(
            [
                loc[:, 0] - box_p[:, 0],
                loc[:, 1] - box_p[:, 1],
                loc[:, 0] + box_p[:, 2],
                loc[:, 1] + box_p[:, 3],
            ],
            1,
        )
        results.append(detections)
    return torch.cat(results, 0)

def make_boxes(location, cls_pred, box_pred, center_pred):
    boxes = []
    for loc, cls_p, box_p, center_p in zip(location, cls_pred, box_pred, center_pred):
        boxes.append(process(loc, cls_p, box_p, center_p))
    return torch.cat(boxes, 0)
        
def compare(p, q):
    cls_pred1, box_pred1, center_pred1, location1 = p
    cls_pred2, box_pred2, center_pred2, location2 = q
    
    
    cls_p1 = flatten(cls_pred1, 11).softmax(-1) ** args.temp
    cls_p2 = flatten(cls_pred2, 11).softmax(-1) ** args.temp
    box_p1 = flatten(box_pred1, 4).relu()
    box_p2 = flatten(box_pred2, 4).relu()
    center_p1 = flatten(center_pred1, 4).sigmoid()
    center_p2 = flatten(center_pred2, 4).sigmoid()
    
    #mask = (cls_p1[:, 1:].max(1)[0].ge(args.mask).float()) * (cls_p2[:, 1:].max(1)[0].ge(args.mask).float())
    
    return (l1loss(cls_p1[:, 1:], cls_p2[:, 1:]), 0, 0, 0)

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

        (loss_dict, _), (loss_dict2, _) = model(images.tensors, targets=targets, r=r)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()
        
        loss = loss_cls + loss_box + loss_center 
        del loss_cls, loss_box, loss_center, loss_dict
        
        loss_cls = loss_dict2['loss_cls'].mean()
        loss_box = loss_dict2['loss_box'].mean()
        loss_center = loss_dict2['loss_center'].mean()
        
        loss += loss_cls + loss_box + loss_center 
        del loss_cls, loss_box, loss_center, loss_dict2
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        c_opt.step()
        g_opt.step()
        
        
        # Train Top
        c_opt.zero_grad()
        (loss_dict, _), (loss_dict2, _) = model(images.tensors, targets=targets, r=r)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()
        
        loss = loss_cls + loss_box + loss_center 
        

        del loss_cls, loss_box, loss_center, loss_dict
        
        loss_cls = loss_dict2['loss_cls'].mean()
        loss_box = loss_dict2['loss_box'].mean()
        loss_center = loss_dict2['loss_center'].mean()
        
        loss += loss_cls + loss_box + loss_center 
        
        loss_reduced = reduce_loss_dict(loss_dict2)
        cls = loss_reduced['loss_cls'].mean().item()
        box = loss_reduced['loss_box'].mean().item()
        center = loss_reduced['loss_center'].mean().item()
        
        del loss_cls, loss_box, loss_center, loss_dict2, loss_reduced

        (loss_dict2, p), (_, q)  = model(target_images.tensors, targets=target_targets, r=r)
        
        cls_discrep, box_discrep, box_mag,_ = compare(p, q)
        dloss = cls_discrep + box_discrep
        
        loss_reduced = reduce_loss_dict(loss_dict2)
        loss_cls_target = loss_reduced['loss_cls'].mean().item()
        loss_box_target = loss_reduced['loss_box'].mean().item()
        loss_center_target = loss_reduced['loss_center'].mean().item()
        
        loss -= dloss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        c_opt.step()
        del loss_dict2, loss_reduced
        
        # Train Bottom
        for j in range(4):
            g_opt.zero_grad()
            '''
            (loss_dict, _), (loss_dict2, _) = model(images.tensors, targets=targets, r=r)
            loss_cls = loss_dict['loss_cls'].mean()
            loss_box = loss_dict['loss_box'].mean()
            loss_center = loss_dict['loss_center'].mean()
            

            loss = loss_cls + loss_box + loss_center
            
            loss_cls2 = loss_dict2['loss_cls'].mean()
            loss_box2 = loss_dict2['loss_box'].mean()
            loss_center2 = loss_dict2['loss_center'].mean()
            loss += loss_cls2 + loss_box2 + loss_center2
            '''
            
            (_, p), (_, q)  = model(target_images.tensors, targets=target_targets, r=r)
            cls_discrep, box_discrep, mask, m = compare(p, q)
            dloss = cls_discrep + box_discrep
            loss = dloss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            g_opt.step()
            
            #del loss_cls, loss_box, loss_center, loss_dict2, loss_dict, loss_cls2, loss_box2, loss_center2
        
        
        discrep_loss = dloss.item()
        cls_discrep, box_discrep, mask = float(cls_discrep), float(box_discrep), float(mask)
        box_discrep = box_discrep if float(m) == 0 else box_discrep / float(m)
        losses.append(cls + box + center)
        dlosses.append(discrep_loss)
        avg = sum(losses) / len(losses)
        davg = sum(dlosses) / len(dlosses)
        
        if i % 100 == 0:
            torch.save((model, c_opt, g_opt), 'slim_fcos_' + str(args.ckpt + epoch + 1) + '.pth')
    
        if get_rank() == 0:
            pbar.set_description(
                (
                    f'epoch: {epoch + 1}; cls: {cls:.4f}; target_cls: {loss_cls_target:.4f};'
                    f'box: {box:.4f}; target_box: {loss_box_target:.4f}; center: {center:.4f}; target_center: {loss_center_target:.4f};'
                    f'cls_discrep: {cls_discrep:.8f}; '
                    f'avg: {avg:.4f}; discrep_avg: {davg:.8f};'
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
    model = nn.DataParallel(model)
    
    bottom = [p for n, p in model.named_parameters() if ('head' not in n)]
    top = [p for n, p in model.named_parameters() if ('head' in n)]

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
        (model, _, _) = torch.load('slim_fcos_' + str(args.ckpt) + '.pth')
        if isinstance(model, nn.DataParallel):
            model = model.module
        model = nn.DataParallel(model)
    else:
        args.ckpt = 0
    for g in c_opt.param_groups:
        g['lr'] = args.lr
    for g in g_opt.param_groups:
        g['lr'] = args.lr2
    model = model.to(device)
    
    for epoch in range(args.epoch):
        train(args, epoch, source_loader, target_loader, model, c_opt, g_opt, device)
        torch.save((model, c_opt, g_opt), 'slim_fcos_' + str(args.ckpt + epoch + 1) + '.pth')
        valid(args, epoch, source_valid_loader, source_valid_set, model, device)
        valid(args, epoch, target_valid_loader, target_valid_set, model, device)
        
        
       
        


