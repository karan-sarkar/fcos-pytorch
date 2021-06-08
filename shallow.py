import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torchvision.models as models
import numpy as np
import math

class Shallow(nn.Module):
    def __init__(self):
        super(Shallow, self, n_classes).__init__()

        backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.num_filters = backbone.fc.in_features
        self.head = nn.Linear(self.num_filters, n_classes)

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.head(features)