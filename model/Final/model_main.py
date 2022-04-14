# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:41:40 2022

@author: ChangGun Choi
"""

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
#%matplotlw32ew3ib inline

from torch import optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from torchvision import utils
from torchvision.datasets import CIFAR10

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
import numpy as np
import time
import copy
import random
from tqdm import tqdm
import math
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# !cd "C:/Users/ChangGun Choi/Team Project/Thesis_Vision/model/Final"
from Patch_embedding import *

#%%
# specify path to data
path2data = 'C:/Users/ChangGun Choi/Team Project/Thesis_Vision/data'

# if not exists the path, make the directory
if not os.path.exists(path2data):
    os.mkdir(path2data)

# load dataset
#train_ds = datasets.STL10(path2data, split='train', download=True, transform=transforms.ToTensor())
#val_ds = datasets.STL10(path2data, split='test', download=True, transform=transforms.ToTensor())
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

train_ds = torchvision.datasets.CIFAR10(root=path2data, train=True,download=True, transform=transform)
train_dl  = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

val_ds = torchvision.datasets.CIFAR10(root=path2data, train=False, download=True, transform=transform)
val_dl  = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#%%

# Testing
in_channels=3
patch_size=16  # x(input 개수)
emb_size=768
img_size=256
x = torch.randn(16, 3, 224, 224).to(device) # batchsize: 16

projection = nn.Sequential(
    # using a conv layer instead of a linear one -> performance gains
    nn.Conv2d(in_channels, emb_size, patch_size, stride=patch_size), # filter가 patchsize
    Rearrange('b e (h) (w) -> b (h w) e') # e: dimension             # 안겹치게 stride
).to(device)

cls_token = nn.Parameter(torch.randn(1,1,emb_size)).to(device)
positions = nn.Parameter(torch.randn((img_size // patch_size) * (img_size // patch_size) + 1, emb_size)).to(device)
batch_size = 16
cls_tokens = repeat(cls_token, '() n e -> b n e', b=batch_size)

print(projection(x).size()) #torch.Size([16, 196, 768])  # 패치갯수: 14 * 14 = 196
print(cls_token.size())  #torch.Size([1, 1, 768])  # 랜덤이여서 배워야하는 parameter
print(cls_tokens.shape)  #torch.Size([16, 1, 768])
print(positions.size())  #torch.Size([197, 768])   # 배워야하는 parameter

patch_embedding = PatchEmbedding().to(device)
patch_output = patch_embedding(x)
print('[batch, 1+num of patches, emb_size] = ', patch_output.shape) # torch.Size([16, 197, 768])


















