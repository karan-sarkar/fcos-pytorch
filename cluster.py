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



from bdd import *

CLUSTERS = int(input("clusters"))
BATCH_SIZE = int(input("batch"))

root_img_path = "bdd100k/images/100k"
root_anno_path = "bdd100k_labels/labels"

train_img_path = root_img_path + "/train/"
val_img_path = root_img_path + "/val/"

train_anno_json_path = root_anno_path + "/bdd100k_labels_images_train.json"
val_anno_json_path = root_anno_path + "/bdd100k_labels_images_val.json"

with open(train_anno_json_path, "r") as file:
    train_data = json.load(file)
print(len(train_data))
with open(val_anno_json_path, "r") as file:
    test_data = json.load(file)
print(len(test_data))

def make_dataset(train):
    if train:
        data = train_data
        json_file = train_anno_json_path
        header = train_img_path
    else:
        data = test_data
        json_file = val_anno_json_path
        header = val_img_path
    
    img_list = []
    for i in tqdm.tqdm(range(len(data))):
        img_list.append(header + data[i]['name'])
    dset = BDD(img_list, json_file)
    return dset

train_dat = make_dataset(True)
val_dat = make_dataset(False)

def load(dset):
    return torch.utils.data.DataLoader(dset,batch_size=BATCH_SIZE,shuffle=True, collate_fn=dset.collate_fn)


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
    result = torch.cat(result, 1)
    return result / result.size(1)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = Backbone()
FEATURES = backbone.num_filters
backbone = backbone.to(device)
means = None


def train(dataset, model, means):
    i = 0
    loss = []
    
    pbar = tqdm.tqdm(load(dataset))
    results = []
    with torch.no_grad():
        for images, boxes, labels, attr in pbar:
            images = images.to(device)
            features = style(model(images))
            if means is None:
                means = torch.randn(CLUSTERS, features.size(1)).to(device)
            
            x2 = (features * features).sum(1).view(-1, 1)
            y2 = (means * means).sum(1).view(1, -1)
            xy = torch.einsum('bf,kf->bk', features, means)
            dist = -2 * xy + x2 + y2
            
            loss.append(float(dist.min(1)[0].mean()))
            avg = sum(loss) / len(loss)
            pbar.set_description(str(avg))
            
            
            
            clusters = F.one_hot(dist.argmin(1), CLUSTERS).float()
            change = torch.einsum('bf,bk->kf', features, clusters)
            means = means * (i/(i + 1)) + change * (1/(i + 1))
            
            k = 0
            classes = dist.argmin(1)
            for flags in attr:
                for j in range(len(flags)):
                    if j > len(results) - 1:
                        results.append(Counter())
                    results[j][(flags[j], int(classes[k]) )] += 1
                k += 1
            
            del images, features, dist, clusters, change, boxes, labels, attr, x2, y2, xy
            
            i += 1
    
    
    print(results)        

for _ in range(100):
    train(train_dat, backbone, means)