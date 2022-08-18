# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 20:46:15 2022

@author: ChangGun Choi

https://yjs-program.tistory.com/171 ################################
https://www.youtube.com/watch?v=5lFiZTSsp40&t=476s 
cd C:/Users/ChangGun Choi/Team Project/Thesis_Vision/Adversarial_attack
"""
from __future__ import print_function
import os 
os.chdir('C:/Users/ChangGun Choi/Team Project/Thesis_Vision')
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from Adversarial_attack.attacks import *

import numpy as np


from PIL import Image
import json

use_cuda=True
#%%

import matplotlib.pyplot as plt
# ## 데이터셋 불러오기
CLASSES = json.load(open('Adversarial_attack/imagenet_classes.json'))
len(CLASSES)
idx2class = [CLASSES[str(i)] for i in range(1000)]
idx2class

# 이미지 불러오기
img = Image.open('Adversarial_attack/corgie.jpg')
img_transforms = transforms.Compose([
    transforms.Resize((224, 224), Image.BICUBIC),
    transforms.ToTensor(),])
img_tensor = img_transforms(img)
img_tensor = img_tensor.unsqueeze(0)
print("이미지 텐서 모양:", img_tensor.size())
# 시각화를 위해 넘파이 행렬 변환
original_img_view = img_tensor.squeeze(0).detach()  # [1, 3, 244, 244] -> [3, 244, 244]
original_img_view = original_img_view.transpose(0,2).transpose(0,1).numpy()
# 텐서 시각화
plt.imshow(original_img_view)

#%% 
"Model"
# ## 공격 전 성능 확인하기
#models.resnet18()
#resnet18 = models.resnet101(pretrained=True)# Model로 ResNet-101버전 로딩
#efficientnet_b7 = models.efficientnet_b7(pretrained=True)
#models.efficientnet_b7
#efficientnet_b6 = models.efficientnet_b6(pretrained=True)
#model = resnet18
#model.eval()
[ 'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',]
#%%
#deit_small_distilled_patch16_224
#model = torch.hub.load('facebookresearch/deit:main','deit_small_distilled_patch16_224')
model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True).eval().to(device)

# 4번째 부터 attack 성공 


model = model.eval()
print(model)#모델 구조 프린트

output = model(img_tensor)
prediction = output.max(1, keepdim=False)[1]   #tensor([263]) Index
prediction_idx = prediction.item() # item 은 숫자만 # 263

prediction_name = idx2class[prediction_idx] 

print("예측된 레이블 번호:", prediction_idx) 
print("레이블 이름:", prediction_name)  # Pembroke, Pembroke Welsh corgi

#%%  적대적 예제 생성
# 이미지의 기울기값을 구하도록 설정
img_tensor.requires_grad_(True)

# 이미지를 모델에 통과시킴
output = model(img_tensor)
# 오차값 구하기 (레이블 263은 웰시코기)
loss = F.nll_loss(output, torch.tensor([263])) 

# 기울기값 구하기
model.zero_grad()
loss.backward()

#%% 1) fgsm_attack

def softmax_activation(inputs): 
    inputs = inputs.tolist()
    exp_values = np.exp(inputs - np.max(inputs)) 
    
    # Normalize 
    probabilities = exp_values / np.sum(exp_values)
    return probabilities 
#%%
epsilons = [0, 0.1/255, 0.3/255, 1/255, 4/255] 
# 이미지의 기울기값을 추출
gradient = img_tensor.grad.data
# FGSM 공격으로 적대적 예제 생성
for epsilon in epsilons:
    perturbed_data = fgsm_attack(img_tensor, epsilon, gradient)
    
    # 생성된 적대적 예제를 모델에 통과시킴
    output = model(perturbed_data)
    #output[:,263]
    # ## 적대적 예제 성능 확인
    
    perturbed_prediction = output.max(1, keepdim=True)[1]
    perturbed_prediction_idx = perturbed_prediction.item()
    perturbed_prediction_name = idx2class[perturbed_prediction_idx]
    #print(output[:,263])
    accuracy = np.max(softmax_activation(output), axis=1)
    accuracy = round(accuracy[0], 2)
    print("Accuracy on benign examples: {}%".format(accuracy * 100)) 
    print("Predicted label:", perturbed_prediction_idx)  # 예측된 레이블 번호: 172
    print("Predicted label name:", perturbed_prediction_name)   # 레이블 이름: whippet
#%% Visualize 

# 시각화를 위해 넘파이 행렬 변환
perturbed_data_view = perturbed_data.squeeze(0).detach()
perturbed_data_view = perturbed_data_view.transpose(0,2).transpose(0,1).numpy()

plt.imshow(perturbed_data_view)

# ## 원본과 적대적 예제 비교
f, a = plt.subplots(1, 2, figsize=(10, 10))

# 원본
a[0].set_title(prediction_name)
a[0].imshow(original_img_view)

# 적대적 예제
a[1].set_title(perturbed_prediction_name)
a[1].imshow(perturbed_data_view)

plt.show()

#%% Test






















