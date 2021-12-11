import math

import torch
from torch import nn
from torch.nn import functional as F

from torch import linalg as LA
import numpy as np

from loss import FCOSLoss
from postprocess import FCOSPostprocessor
from boxlist import BoxList


    
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
        
        self.backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]))
        
        self.fc1 = nn.Linear(2048, 4096) 
        self.fc2 = nn.Linear(4096, 2048)
        self.size = config.n_class -1 + 1 * 4
        self.limit = 64
        
        self.enc1 = nn.Linear(self.size, self.limit)
        self.enc2 = nn.Linear(self.limit, self.limit)
        self.dec1 = nn.Linear(self.limit, self.limit)
        self.dec2 = nn.Linear(self.limit, self.size)
        
        
        self.fc3 = nn.Linear(2048, self.size * self.size)
        self.fc4 = nn.Linear(2048, self.limit * self.limit)
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
        #print(features.shape)
        features = features.view(features.size(0), -1)
        matrix = (self.fc1(features.relu()))
        matrix = self.fc2(matrix.relu())
        A, B = self.fc3(matrix.relu()), self.fc4(matrix.relu())
        A = A.view(A.size(0), self.size, self.size)
        B = B.view(B.size(0), self.limit, self.limit)
        #print(matrix)
        A = torch.einsum('bcd,bde->bce', A, A)
        B = torch.einsum('bcd,bde->bce', B, B)
        
        #print([t.box.shape for t in targets])
        if self.training:
            boxes = [t.box/500 for t in targets if t.box.numel() > 0]
            #print(boxes[0], boxes[0].shape)
            labels = [F.one_hot(t.fields['labels'] - 1, self.config.n_class - 1).float()*3 for t in targets if t.box.numel() > 0]
            #print(labels[0], labels[0].shape)
            vectors = [torch.cat([b, l], -1)[:self.limit] for b, l in zip(boxes, labels)]
            old_vectors = vectors
            vectors = [self.enc2(self.enc1(v).relu()) for v in vectors]
            rec_vectors = [self.dec2(self.dec1(v.relu()).relu()) for v in vectors]
            loss_rec = torch.stack([self.crit(o,r) for o, r in zip(old_vectors, rec_vectors)], 0).mean()
            
            #print(vectors[0], vectors[.shape)
            del boxes, labels
            
            svd = [LA.svd(m, full_matrices=False) for m in vectors]
            
            d = A.device
            #print([(u.shape, s.shape, vh.shape) for u, s, vh in svd])
            #svd = [(u.to(d), s.to(d), vh.to(d), max(u.size(0), vh.size(0))) for u,s,vh in svd]
            svd = [(pad(u, (self.limit, self.limit)), pad(torch.diag(s), (self.limit, self.limit)), pad(vh, (self.limit, self.limit))) for (u, s, vh) in svd]
            #print([(vh.shape) for u, s, vh in svd])
            U, P = zip(*[(u @ pad(vh, (self.limit, self.limit)), vh.transpose(0, 1) @ s @ vh) for u, s, vh in svd])
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
                'loss_herm': loss_herm,
                'loss_unit': loss_unit,
                'loss_rec': loss_rec
            }
            
            
            return None, losses
        
        else:
            #print(matrix.shape)
            #print(P.shape, A.shape)
            
            
            w, v = LA.eigh(B)
            v = torch.einsum('bcd,bde->bce', v.transpose(-1, 2), A).abs()
            v = v.view(-1, self.limit)
            v = self.dec2(self.dec1(v.relu()).relu())
            v = v.view(-1, self.limit, self.limit)
            #print(v)           
            
            #print(w.sort(-1))
            #print(v, '\n\n')

            b = v[:, :, :4] * 500
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
        
        
        
