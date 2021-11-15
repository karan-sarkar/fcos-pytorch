import os
import gc
from collections import Counter


import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler


from tqdm import tqdm


from argument import get_args
from backbone import vovnet57, vovnet39, resnet50
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


def train(args, epoch, loader, target_loader, model, g_optimizer, l_optimizer, d_optimizer, device):
    model.train()
    
   
    memory = Counter()

    pbar = tqdm(range(min(len(loader), len(target_loader))))
    iterator = iter(loader)
    target_iterator = iter(target_loader)


    for i in pbar:
        try:
            images, targets, _ = next(iterator)
            target_images, target_targets, _ = next(target_iterator)
        except:
            iterator = iter(loader)
            target_iterator = iter(target_loader)
            
            images, targets, _ = next(iterator)
            target_images, target_targets, _ = next(target_iterator)

        
        model.zero_grad()

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        target_images = target_images.to(device)
        target_targets = [target.to(device) for target in target_targets]


        _, loss_dict = model(images.tensors, targets=targets)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()
        
        del loss_dict

        loss = loss_cls + loss_box + loss_center
        del loss_cls, loss_box, loss_center


        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        g_optimizer.step()
        d_optimizer.step()
        l_optimizer.step()

        del loss, _


        model.zero_grad()

        _, loss_dict = model(images.tensors, targets=targets)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()
        loss_discrep = loss_dict['loss_discrep'].mean()
        
        loss = loss_cls + loss_box + loss_center + loss_discrep
        del loss_cls, loss_box, loss_center, loss_discrep

        loss_reduced = reduce_loss_dict(loss_dict)
        loss_cls_item = loss_reduced['loss_cls'].mean().item()
        loss_box_item = loss_reduced['loss_box'].mean().item()
        loss_center_item = loss_reduced['loss_center'].mean().item()
        loss_discrep_item = loss_reduced['loss_discrep'].mean().item()
        
        del loss_dict, loss_reduced

        _, target_loss_dict = model(target_images.tensors, targets=target_targets)
        target_loss_discrep = target_loss_dict['loss_discrep'].mean()
        del target_loss_dict


        loss -= target_loss_discrep
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        d_optimizer.step()


        del loss, target_loss_discrep, _


        for _ in range(4):
            model.zero_grad()
            
            _, loss_dict = model(images.tensors, targets=targets)
            loss_cls = loss_dict['loss_cls'].mean()
            loss_box = loss_dict['loss_box'].mean()
            loss_center = loss_dict['loss_center'].mean()
            del loss_dict

            loss = loss_cls + loss_box + loss_center
            del loss_cls, loss_box, loss_center

            _, target_loss_dict = model(target_images.tensors, targets=target_targets)
            target_loss_discrep = target_loss_dict['loss_discrep'].mean()
            del target_loss_dict

            loss += target_loss_discrep
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            l_optimizer.step()
            g_optimizer.step()

            del loss, target_loss_discrep, _
        
        del images, targets, target_images, target_targets


        
        pbar.set_description(
            (
                f'epoch: {epoch + 1}; cls: {loss_cls_item:.4f}; '
                f'box: {loss_box_item:.4f}; center: {loss_center_item:.4f}; discrep: {loss_discrep_item:.4f}'
            )
        )
        
        '''
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print(t, r, a)
        
        new_memory = Counter()
        values = {}
        for obj in gc.get_objects():
            try:
                new_memory[(type(obj))] += 1
            except:
                pass
        for obj in gc.get_objects():
            if new_memory[(type(obj))] != memory[(type(obj))]:
                if (type(obj)) not in values:
                    values[(type(obj))] = []
                values[(type(obj))].append(obj) 
        for mem in new_memory.keys():
            if memory[mem] != new_memory[mem]:
                print(new_memory[mem] - memory[mem], mem)
        del memory
        memory = new_memory
        del new_memory
        '''
        

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

    train_set = COCODataset(args.path, args.domain, 'train', preset_transform(args, train=True))
    valid_set = COCODataset(args.path, args.domain, 'val', preset_transform(args, train=False))

    target_train_set = COCODataset(args.path, args.target_domain, 'train', preset_transform(args, train=True))
    target_valid_set = COCODataset(args.path, args.target_domain, 'val', preset_transform(args, train=False))



    backbone = resnet50(pretrained=True,if_include_top=False)
    model = FCOS(args, backbone)
    model = model.to(device)

    g_params = [p for n,p in model.named_parameters() if ('discriminator' not in n and 'head' not in n)]
    l_params = [p for n,p in model.named_parameters() if 'head' in n]
    d_params = [p for n,p in model.named_parameters() if 'discriminator' in n]
    print(len(g_params), len(l_params), len(d_params))


    g_optimizer = optim.SGD(
        g_params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.l2,
        nesterov=True,
    )

    l_optimizer = optim.SGD(
        l_params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.l2,
        nesterov=True,
    )

    d_optimizer = optim.SGD(
        d_params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.l2,
        nesterov=True,
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        g_optimizer, milestones=[16, 22], gamma=0.1
    )

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        sampler=data_sampler(train_set, shuffle=True, distributed=args.distributed),
        num_workers=2,
        collate_fn=collate_fn(args),
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch,
        sampler=data_sampler(valid_set, shuffle=False, distributed=args.distributed),
        num_workers=2,
        collate_fn=collate_fn(args),
    )


    target_train_loader = DataLoader(
        target_train_set,
        batch_size=args.batch,
        sampler=data_sampler(target_train_set, shuffle=True, distributed=args.distributed),
        num_workers=2,
        collate_fn=collate_fn(args),
    )
    target_valid_loader = DataLoader(
        target_valid_set,
        batch_size=args.batch,
        sampler=data_sampler(valid_set, shuffle=False, distributed=args.distributed),
        num_workers=2,
        collate_fn=collate_fn(args),
    )


    for epoch in range(args.epoch):
        train(args, epoch, train_loader, target_train_loader, model, g_optimizer, l_optimizer, d_optimizer, device)
        valid(args, epoch, valid_loader, valid_set, model, device)
        valid(args, epoch, target_valid_loader, target_valid_set, model, device)

        scheduler.step()

        if get_rank() == 0:
            torch.save(
                {'model': model.module.state_dict(), 'g_optim': g_optimizer.state_dict(),
                'l_optim': l_optimizer.state_dict(), 'd_optim': d_optimizer.state_dict()},
                f'checkpoint/adapt-epoch-{epoch + 1}.pt',
            )

