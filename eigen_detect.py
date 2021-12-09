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


class EigenDetect(nn.Module):
    def __init__(self, config, backbone):
        super().__init__()

        self.backbone = backbone
        fpn_top = FPNTopP6P7(
            config.feat_channels[-1], config.out_channel, use_p5=config.use_p5
        )
        self.fpn = FPN(config.feat_channels, config.out_channel, fpn_top)
        self.fc = nn.Linear(config.out_channel * 5033, 2 * 54 * 54) 
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
        matrix = (self.fc(features.view(features.size(0), -1).relu()) / (2 * 54)).tanh() 
        matrix = matrix.view(matrix.size(0), 2, 54, 54)
        #print(matrix)
        matrix = torch.einsum('bncd->bnde->bnce', matrix, matrix)
        
        #print([t.box.shape for t in targets])
        if self.training:
            boxes = [binary(t.box.int().view(-1), 11).float().view(-1, 4 * 11) for t in targets if t.box.numel() > 0]
            #print(boxes[0], boxes[0].shape)
            labels = [F.one_hot(t.fields['labels'] - 1, self.config.n_class - 1).float() for t in targets if t.box.numel() > 0]
            #print(labels[0], labels[0].shape)
            vectors = [torch.cat([b, l], -1) for b, l in zip(boxes, labels)]
            #print(vectors[0], vectors[0].shape)
            del boxes, labels
            
            svd = [torch.linalg.svd(m, full_matrices=False) for m in vectors]
            
            d = matrix.device
            svd = [torch.cat([u, torch.zeros(u.size(0), vh.size(0) - u.size(0)).to(d)], 1), torch.cat([s, torch.zeros(vh.size(0) - s.size(0), s.size(0)).to(d)], 0), v for u, s, v in svd]
            U, P = zip(*[u @ vh, vh.transpose() @ s @ v for u, s, vh in svd])
            print(vectors[0], '\n\n\n')
            print(U[0] @ P[0])
            P = torch.stack(p, 0)
            
            loss_herm = self.crit(matrix[:, 0], P)
            D = [U[i] @ matrix[i, 1] @ U[i].transpose() for i in range(len(U))]
            loss_unit = torch.stack([self.crit(D, torch.zeros_like(D)) / torch.trace(D @ D) for d in D], 0).mean()

            losses = {
                'loss_pos': loss_herm,
                'loss_neg': loss_unit,
            }
            
            
            return None, losses
        
        else:
            P = matrix[:, 0]
            A = matrix[:, 1]
            
            
            w, v = LA.eigh(A.detach().cpu().numpy())
            w = w.transpose() @ P
            
            #print(w.sort(-1))
            #print(v, '\n\n')

            b = v[:, :44, :].transpose(-1, -2)
            l = v[:, 44:, :].transpose(-1, -2)

            b = decimal(b.ge(0).reshape(-1, 54, 4, 11), 11)
            l = l.argmax(-1)

            #print(b.shape, l.shape, w.shape)

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
        
        
        
