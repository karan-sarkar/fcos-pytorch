import math

import torch
from torch import nn
from torch.nn import functional as F

from torch import linalg as LA
import numpy as np

from loss import FCOSLoss
from postprocess import FCOSPostprocessor
from boxlist import BoxList

def init_conv_kaiming(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight, a=1)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class FPN(nn.Module):
    def __init__(self, in_channels, out_channel, top_blocks=None):
        super().__init__()

        self.inner_convs = nn.ModuleList()
        self.out_convs = nn.ModuleList()
        #print(in_channels)
        for i, in_channel in enumerate(in_channels[1:], 1):
            if in_channel == 0:
                self.inner_convs.append(None)
                self.out_convs.append(None)

                continue

            inner_conv = nn.Conv2d(in_channel, out_channel, 1)
            feat_conv = nn.Conv2d(out_channel, out_channel, 3, padding=1)

            self.inner_convs.append(inner_conv)
            self.out_convs.append(feat_conv)

        self.apply(init_conv_kaiming)

        self.top_blocks = top_blocks

    def forward(self, inputs):
        inner = self.inner_convs[-1](inputs[-1])
        outs = [self.out_convs[-1](inner)]

        for feat, inner_conv, out_conv in zip(
            inputs[:-1][::-1], self.inner_convs[:-1][::-1], self.out_convs[:-1][::-1]
        ):
            if inner_conv is None:
                continue

            upsample = F.interpolate(inner, scale_factor=2, mode='nearest')
            inner_feat = inner_conv(feat)
            inner = inner_feat + upsample
            outs.insert(0, out_conv(inner))

        if self.top_blocks is not None:
            top_outs = self.top_blocks(outs[-1], inputs[-1])
            outs.extend(top_outs)

        return outs

class FPNTopP6P7(nn.Module):
    def __init__(self, in_channel, out_channel, use_p5=True):
        super().__init__()

        self.p6 = nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1)
        self.p7 = nn.Conv2d(out_channel, out_channel, 3, stride=2, padding=1)

        self.apply(init_conv_kaiming)

        self.use_p5 = use_p5

    def forward(self, f5, p5):
        input = p5 if self.use_p5 else f5

        p6 = self.p6(input)
        p7 = self.p7(F.relu(p6))

        return p6, p7
    
def binary(x, bits):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    mask = mask.repeat(x.size(0), 1)
    y = x.unsqueeze(-1).repeat(1, mask.size(-1))
    #print(mask.shape, y.shape)
    return 2*y.bitwise_and(mask).ne(0).float() - 1

def decimal(x, bits):
     mask = 2**torch.arange(bits).to(x.device)
     #print(torch.arange(bits))
     #print(mask)
     mask = mask.repeat(*x.shape[:-1], 1)
     #print(mask, '\n\n')
     #print(x.int() * mask)
     return (x.int() * mask).sum(-1)
    
def pad(x, sh):
    #print(x.shape, sh)
    val = torch.zeros(sh).to(x.device)
    val[:min(x.size(0), sh[0]), :min(x.size(1), sh[1])] = x
    #print(val.shape)
    return val


class EigenDetect(nn.Module):
    def __init__(self, config, backbone):
        super().__init__()

        self.backbone = backbone
        fpn_top = FPNTopP6P7(
            config.feat_channels[-1], config.out_channel, use_p5=config.use_p5
        )
        self.fpn = FPN(config.feat_channels, config.out_channel, fpn_top)
        self.fc1 = nn.Linear(config.out_channel * 5033, 4000) 
        self.fc2 = nn.Linear(4000, 2000)
        self.size = config.n_class -1 + 1 * 4
        self.limit = self.size
        self.fc3 = nn.Linear(2000, self.size * self.size)
        self.fc4 = nn.Linear(2000, self.limit * self.limit)
        self.config = config
        self.crit = nn.MSELoss()
    
    def train(self, mode=True):
        super().train(mode)

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

        self.apply(freeze_bn)
    
    def forward(self, input, image_sizes=None, targets=None):
        features = self.backbone(input)
        #print([f.shape for f in features])
        #features = torch.stack(features, 0)
        features = self.fpn(features)
        #print([f.shape for f in features])
        features = torch.cat([f.view(f.size(0), f.size(1), -1) for f in features], -1)
        #print(features.shape)
        matrix = (self.fc1(features.view(features.size(0), -1).relu()))
        matrix = self.fc2(matrix.relu())
        A, B = self.fc3(matrix.relu()), self.fc4(matrix.relu())
        A = A.view(A.size(0), self.size, self.size)
        B = B.view(B.size(0), self.limit, self.limit)
        #print(matrix)
        A = torch.einsum('bcd,bde->bce', A, A)
        B = torch.einsum('bcd,bde->bce', B, B)
        
        #print([t.box.shape for t in targets])
        if self.training:
            boxes = [t.box/200 for t in targets if t.box.numel() > 0]
            #print(boxes[0], boxes[0].shape)
            labels = [F.one_hot(t.fields['labels'] - 1, self.config.n_class - 1).float()*3 for t in targets if t.box.numel() > 0]
            #print(labels[0], labels[0].shape)
            vectors = [torch.cat([b, l], -1)[:self.limit] for b, l in zip(boxes, labels)]
            #print(vectors[0], vectors[.shape)
            del boxes, labels
            
            svd = [LA.svd(m, full_matrices=False) for m in vectors]
            
            d = A.device
            #print([(u.shape, s.shape, vh.shape) for u, s, vh in svd])
            #svd = [(u.to(d), s.to(d), vh.to(d), max(u.size(0), vh.size(0))) for u,s,vh in svd]
            svd = [(pad(u, (self.limit, self.limit)), pad(torch.diag(s), (self.limit, self.size)), pad(vh, (self.size, self.size))) for (u, s, vh) in svd]
            #print([(vh.shape) for u, s, vh in svd])
            U, P = zip(*[(u @ pad(vh, (self.limit, self.size)), vh.transpose(0, 1) @ s @ vh) for u, s, vh in svd])
            #print(vectors[0], '\n\n')
            #print(U[0] @ P[0])
            P = torch.stack(P, 0)
            
            loss_herm = self.crit(A, P)
            
            D = [U[i] @ B[i] @ U[i].transpose(0,1) for i in range(len(U))]
            #print(D[0])
            loss_unit = torch.stack([self.crit(d, torch.zeros_like(d)) / torch.diag(d).square().mean() for d in D], 0).mean()
            #print(D[0])
            #print(matrix[0, 0, 0, 0], P[0, 0, 0])
            losses = {
                'loss_pos': loss_herm,
                'loss_neg': loss_unit,
            }
            
            
            return None, losses
        
        else:
            #print(matrix.shape)
            #print(P.shape, A.shape)
            
            
            w, v = LA.eigh(B)
            v = torch.einsum('bcd,bde->bce', v.transpose(-1, 2), A).abs()
            #print(v)           
            
            #print(w.sort(-1))
            #print(v, '\n\n')

            b = v[:, :, :4] * 200
            l = v[:, :, :(self.config.n_class - 1)]

            l = l.argmax(-1)
            print(b.shape, l.shape)
            print(b[0], l[0]+1)           #print(b.shape, l.shape, w.shape)

            boxes = []
            for i in range(w.size(0)):
                mybox = b[i][w[i] > 0]
                mylabel = l[i][w[i] > 0]
                myscores = w[i][w[i] > 0]
                #print(mybox, mylabel)
                #print(targets[i].box, targets[i].fields['labels'])
                box = BoxList(mybox, image_sizes[i])
                box.fields['labels'] = mylabel + 1
                box.fields['scores'] = myscores
                boxes.append(box)

            return boxes, None
        
        
        
