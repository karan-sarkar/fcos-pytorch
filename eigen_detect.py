import math

import torch
from torch import nn
from torch.nn import functional as F

from loss import FCOSLoss
from postprocess import FCOSPostprocessor

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

        for i, in_channel in enumerate(in_channels, 1):
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
    return y.bitwise_and(mask).ne(0).byte()

class EigenDetect(nn.Module):
    def __init__(self, config, backbone):
        super().__init__()

        self.backbone = backbone
        fpn_top = FPNTopP6P7(
            config.feat_channels[-1], config.out_channel, use_p5=config.use_p5
        )
        self.fpn = FPN(config.feat_channels, config.out_channel, fpn_top)
        self.fc = nn.Linear(config.out_channel * 5033, 54 * 54) 
        self.config = config
        self.crit = nn.BCEWithLogitsLoss()
    
    def train(self, mode=True):
        super().train(mode)

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

        self.apply(freeze_bn)
    
    def forward(self, input, image_sizes=None, targets=None):
        features = self.backbone(input)[1:]
        #print([f.shape for f in features])
        #features = torch.stack(features, 0)
        features = self.fpn(features)
        #print([f.shape for f in features])
        features = torch.cat([f.view(f.size(0), f.size(1), -1) for f in features], -1)
        #print(features.shape)
        matrix = self.fc(features.view(features.size(0), -1).relu()).tanh()
        matrix = matrix.view(matrix.size(0), 54, 54)
        
        
        if self.training:
            boxes = [binary(t.box.int().view(-1), 11).float().view(-1, 4 * 11) for t in targets]
            #print(boxes[0], boxes[0].shape)
            labels = [F.one_hot(t.fields['labels'], self.config.n_class - 1) for t in targets]
            #print(labels[0], labels[0].shape)
            vectors = [torch.cat([b, l], -1) for b, l in zip(boxes, labels)]
            #print(vectors[0], vectors[0].shape)
            
            temp = [torch.einsum('nm,vm->nv', matrix[i], vectors[i]) for i in range(len(vectors))]
            pos = [torch.einsum('vn,nv->v', v, t) for v, t in zip(vectors, temp)]
            loss_pos = torch.stack([self.crit(p, torch.ones_like(p)) for p in pos]).mean()
            
            neg_boxes = torch.randint(1, (32, 44)).to(loss_pos.device)
            neg_labels = F.one_hot(torch.randint(self.config.n_class - 1, (32,)), self.config.n_class - 1).to(loss_pos.device)
            neg_vectors = torch.cat([neg_boxes, neg_labels], -1)
            
            temp2 = torch.einsum('bnm,vm->bnv', matrix, neg_vectors)
            neg = torch.einsum('vn,bnv->bv', neg_vectors,temp2)
            loss_neg = self.crit(neg, torch.zeros_like(neg))
            
            
            losses = {
                'loss_pos': loss_pos,
                'loss_neg': loss_neg,
            }
            
            return None, losses
        '''
        else:
            boxes = self.postprocessor(
                location, cls_pred, box_pred, center_pred, image_sizes
            )

            return boxes, None
        '''
        
        
