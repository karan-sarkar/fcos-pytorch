import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import math
import json
import tqdm
import time
from collections import Counter
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import homogeneity_score



from bdd import *

BATCH_SIZE = int(input("batch"))

trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
 ])

train_dat = torchvision.datasets.CIFAR100(root = 'CIFAR100/', train = True, download = True, transform = trans)

def load(dset):
    return torch.utils.data.DataLoader(dset,batch_size=BATCH_SIZE,shuffle=True)


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        backbone = models.resnet18(pretrained=True)
        self.layers = list(backbone.children())[:-1]
        self.layers = [layer.to(device) for layer in self.layers]
        self.num_filters = backbone.fc.in_features

    def forward(self, x):
        y = []
        for layer in self.layers:
            x = layer(x)
            y.append(x)
        return y

def style(layers):
    result = [torch.einsum('bcmn,bdmn->bcd', layer, layer).view(layer.size(0), -1) for layer in layers]
    result = [layer / layer.size(1) for layer in result]
    result = torch.cat(result, 1)
    return result / len(layers)
        
def last(layers):
    result = layers[-1].view(layers[-1].size(0), -1)
    return result / result.size(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Backbone()
FEATURES = model.num_filters
model = model.to(device)


for CLUSTERS in range(1, 11):

    means = None
    counts = None
    iter = 0
    loss = []
    
    print(CLUSTERS)
    
    pbar = tqdm.tqdm(load(train_dat))
    
    labels_true = []
    labels_pred = []
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            features = style(model(images))
            if means is None:
                means = features[:CLUSTERS].to(device)
                counts = torch.zeros(CLUSTERS).to(device)
            
            x2 = (features * features).sum(1).view(-1, 1)
            y2 = (means * means).sum(1).view(1, -1)
            xy = torch.einsum('bf,kf->bk', features, means)
            dist = -2 * xy + x2 + y2
            
            loss.append(float(dist.min(1)[0].mean()))
            avg = sum(loss) / len(loss)
            res = str(avg)
            
            assign = dist.argmin(1)
            clusters = F.one_hot(assign, CLUSTERS).float()
            current = clusters.sum(0)
            counts += current
            
            mean_split = [means[i] for i in range(means.size(0))]
            for i in range(features.size(0)):
                mean_split[int(assign[i])] += (features[i] - mean_split[int(assign[i])]) / counts[int(assign[i])]
            means = torch.stack(mean_split, 0)
            
            labels_true.append(labels.cpu().detach().numpy())
            labels_pred.append(assign.cpu().detach().numpy())
            
            res += ' ' + adjusted_mutual_info_score(np.concatenate(labels_true, 0), np.concatenate(labels_pred, 0))
            pbar.set_description(res)
            
            iter += 1

        print(adjusted_mutual_info_score(np.concatenate(labels_true, 0), np.concatenate(labels_pred, 0)))
        print(homogeneity_score(np.concatenate(labels_true, 0), np.concatenate(labels_pred, 0)))
        
        
    