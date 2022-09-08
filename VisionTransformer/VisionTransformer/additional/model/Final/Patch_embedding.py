# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:35:32 2022

@author: ChangGun Choi
"""

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
#%matplotlw32ew3ib inline

#%%
"1. PatchEmbedding"
#latent vector 사이즈 D 를 모든 레이어에 걸쳐 사용하는데, 이를 통해 패치들을 flatten 시키고 D차원으로 매핑
#https://hongl.tistory.com/235 ,  https://yhkim4504.tistory.com/5

#To handle 2D images, reshape the image into a sequence of flattened 2D patches.
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=256):
        super().__init__()
        self.patch_size = patch_size # 패치 이미지의 가로,세로 길이
        self.patch_num = img_size // patch_size # patch_num**2 이 전체 갯수
        
        # # Method 1: Flatten and FC layer
        # self.projection = nn.Sequential(
        #     Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
        #     nn.Linear(path_size * patch_size * in_channels, emb_size))

        # Method 2: Conv (1차원 Flatten)
        #패치 임베딩 projection 는 CNN feature map으로부터 뽑아낸 패치들에 적용
        #input sequence가 feature map의 spatial dimension을 flatten 시키고 트랜스포머 차원으로 projecting 시킴으로써 나온 것
        self.projection = nn.Sequential(
            
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'))
                         # 학습가능 embedding
                         
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size)) # BERT cls토큰 #(1, 1, 768)
        self.positions = nn.Parameter(torch.randn((self.patch_num)**2 + 1, emb_size))
                                                # 패치 갯수 + cls토큰 만큼 position 만듬
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.projection(x)
        # cls_token을 반복하여 배치사이즈의 크기와 맞춰줌
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=batch_size) #(16, 1, 768)
        # cls_token과 projected_x를 concatenate
        x = torch.cat([cls_tokens, x], dim=1)  # 196 + 1 = 197 # dim=1은 가로 row로 붙이는 것: dimension이 늘어나야하거는 니까
        # add position embedding to prejected patches
        x += self.positions  # 자동으로 크기를 맞춰서 연산을 수행하게 만듬: Broadcasting
        
        return x   
    
