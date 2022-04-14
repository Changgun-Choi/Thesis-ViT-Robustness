# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 11:08:32 2022

@author: ChangGun Choi
"""

   class Normalize(nn.Module):
       def __init__(self, mean, std) :
           super(Normalize, self).__init__()
           self.register_buffer('mean', torch.Tensor(mean))
           self.register_buffer('std', torch.Tensor(std))
           
       def forward(self, input):
           # Broadcasting
           mean = self.mean.reshape(1, 3, 1, 1)
           std = self.std.reshape(1, 3, 1, 1)
           return (input - mean) / std
   # We can't use torch.transforms because it supports only non-batch images.
   norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   