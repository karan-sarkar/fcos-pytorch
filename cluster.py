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

CLUSTERS = 10
BATCH_SIZE = 4

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
        img_list.append(header + data[i]['videoName'] + '.jpg')
    dset = BDD(img_list, json_file)
    return dset

train_dat = make_dataset(True)
val_dat = make_dataset(False)

def load(dset, sample):
    return torch.utils.data.DataLoader(dset,batch_size=BATCH_SIZE,shuffle=True, collate_fn=dset.collate_fn)


class Backbone(nn.Modul):
    def __init__(self):
        super(Backbone, self).__init__()

        backbone = models.resnet50(pretrained=True)
        layers = list(backbone.children())[:-1]
        self.num_filters = backbone.fc.in_features
        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x):
        self.feature_extractor.eval()
        return self.feature_extractor(x).flatten(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = Backbone()
FEATURES = backbone.num_filters
backbone = backbone.to(device)
means = torch.randn(CLUSTERS, FEATURES).to(device)


def train(dataset, model, means):
    i = 0
    for images, boxes, labels, attr in tqdm.tqdm(load(dataset)):
        images = images.to(device)
        features = model(images)
        
        x2 = (features * features).sum(1).view(-1, 1)
        y2 = (means * means).sum(1).view(1, -1)
        xy = torch.einsum('bf,kf->bk', features, means)
        dist = -2 * xy + x2 + y2
        
        clusters = F.one_hot(dist.argmin(1), CLUSTERS)
        change = torch.einsum('bf,bk->kf', features, clusters)
        means = means * (i/(i + 1)) + change * (1/(i + 1))
        
        i += 1

def valid(dataset, model, means):
    i = 0
    results = []
    for images, boxes, labels, attr in tqdm.tqdm(load(dataset)):
        images = images.to(device)
        features = model(images)
        
        x2 = (features * features).sum(1).view(-1, 1)
        y2 = (means * means).sum(1).view(1, -1)
        xy = torch.einsum('bf,kf->bk', features, means)
        dist = -2 * xy + x2 + y2
        
        clusters = dist.argmin(1)
        for flags in attr:
            for i in range(len(flags)):
                if i > len(results) - 1:
                    results.append(Counter())
                results[i][flags[i]] += 1
                results[i]['total'] += 1
        
        for res in results:
            total = res['total']
            for key in res.keys():
                if key != 'total':
                    res[key] /= total
        
        print(results)
        
        i += 1

for _ in range(100):
    train(train_dat, backbone, means)
    valid(val_dat, backbone, means)
