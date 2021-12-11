import os
import gc
from collections import Counter


import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
from torchvision import models


from tqdm import tqdm


from argument import get_args
from backbone import vovnet57, vovnet39, resnet50
from dataset import COCODataset, collate_fn
from model import FCOS
from eigen_detect import EigenDetect
from transform import preset_transform
from evaluate import evaluate
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    DistributedSampler,
    all_gather,
)



from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs')
global_iter = 0

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
        print((targets[0].box, targets[0].fields['labels']))
        pred, _ = model(images.tensors, images.sizes, targets)
        pred = [p.to('cpu') for p in pred]

        preds.update({id: p for id, p in zip(ids, pred)})

    preds = accumulate_predictions(preds)
    
    if get_rank() != 0:
        return

    evaluate(dataset, preds)
    return


def train(args, epoch, loader, target_loader, model, optimizer, device):
    model.train()
    global global_iter
   
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

        global_iter += 1
        model.zero_grad()

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        target_images = target_images.to(device)
        target_targets = [target.to(device) for target in target_targets]


        _, loss_dict = model(images.tensors, targets=targets)
        loss_pos = loss_dict['loss_pos'].mean()
        loss_neg = loss_dict['loss_neg'].mean()
        

        loss = loss_pos + loss_neg
        del loss_pos, loss_neg


        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is None:
                print(name)



        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        del loss, _

        
        loss_reduced = reduce_loss_dict(loss_dict)
        loss_pos_item = loss_reduced['loss_pos'].mean().item()
        loss_neg_item = loss_reduced['loss_neg'].mean().item()
        total_loss = loss_pos_item + loss_neg_item
        del loss_dict, loss_reduced
        
        del images, targets, target_images, target_targets


        writer.add_scalar('loss/pos', loss_pos_item, global_iter)
        writer.add_scalar('loss/neg', loss_neg_item, global_iter)
        writer.add_scalar('loss/tot', total_loss, global_iter)
        
        pbar.set_description(
            (
                f'epoch: {epoch + 1}; pos: {loss_pos_item:.4f}; '
                f'neg: {loss_neg_item:.4f}'
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



    backbone = models.resnet152(pretrained=True)
    model = EigenDetect(args, backbone)
    


    model = model.to(device)


    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.l2,
        nesterov=True,
    )


    if args.ckpt is not None:
        mapping = torch.load(args.ckpt)
        model.load_state_dict(mapping['model'])
        optimizer.load_state_dict(mapping['optim'])

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[16, 22], gamma=0.1
    )

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False
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
        train(args, epoch, train_loader, target_train_loader, model, optimizer, device)
        valid(args, epoch, valid_loader, valid_set, model, device)
        valid(args, epoch, target_valid_loader, target_valid_set, model, device)

        scheduler.step()

        if get_rank() == 0:
            torch.save(
                {'model': model.module.state_dict(), 'optim': optimizer.state_dict()},
                f'checkpoint/adapt-epoch-{epoch//5 + 1}.pt'
            )

