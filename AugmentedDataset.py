import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

from randaugment import RandAugment

import tqdm
import itertools
import random

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, data, aug_data, sample):
        super(AugmentedDataset, self).__init__()
        self.data = data
        self.aug_data = aug_data
        self.sample = sample
    
    def __len__(self):
        return self.sample.shape[0]
        
    def __getitem__(self, idx):
        i = self.sample[idx]
        print([type(a) for a in self.data[i]])
        print([type(a) for a in self.aug_data[i]])
        return (self.data[i][0], self.aug_data[i][0], self.data[i][1])

    