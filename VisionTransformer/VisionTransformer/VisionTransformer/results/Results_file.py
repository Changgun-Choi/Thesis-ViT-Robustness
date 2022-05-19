# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:14:40 2022

@author: ChangGun Choi
"""

#%%
# https://colab.research.google.com/drive/1muZ4QFgVfwALgqmrfOkp7trAvqDemckO?usp=sharing
"Test data: 800 dataset"

"FGSM : Resnet"
#clean accuracy:  68.4 %
#robust accuracy for perturbations with
 # Linf norm ≤ 0     : 68.4 %
 # Linf norm ≤ 0.0003921568627450981: 50.6 %
 # Linf norm ≤ 0.001176470588235294: 28.6 %
#  Linf norm ≤ 0.00392156862745098:  4.2 %
 # Linf norm ≤ 0.01568627450980392:  0.5 %
 
"FGSM:EfficientNet"
#robust accuracy for perturbations with
 # Linf norm ≤ 0     : 80.4 %
 # Linf norm ≤ 0.0003921568627450981: 70.5 %
  #Linf norm ≤ 0.001176470588235294: 53.9 %
  #Linf norm ≤ 0.00392156862745098: 33.8 %
  #Linf norm ≤ 0.01568627450980392: 24.5 %
  
"FGSM: ViT"
#clean accuracy:  88.6 %
#robust accuracy for perturbations with
#  Linf norm ≤ 0     : 88.6 %
 # Linf norm ≤ 0.0003921568627450981: 78.7 %
 # Linf norm ≤ 0.001176470588235294: 64.3 %
 # Linf norm ≤ 0.00392156862745098: 42.0 %
 # Linf norm ≤ 0.01568627450980392: 26.7 %

"FGSM: DeiT "    
" Training data-efficient image transformers & distillation through attention"
#robust accuracy for perturbations with
#  Linf norm ≤ 0     : 86.6 %
 # Linf norm ≤ 0.0003921568627450981: 75.6 %
  #Linf norm ≤ 0.001176470588235294: 65.8 %
  #Linf norm ≤ 0.00392156862745098: 55.3 %
  #Linf norm ≤ 0.01568627450980392: 43.5 %
    
"FGSM: Swin"   "Robust to FGSM" 
clean accuracy:  87.6 %
#robust accuracy for perturbations with
 # Linf norm ≤ 0     : 87.6 %
  #Linf norm ≤ 0.0003921568627450981: 76.6 %
  #Linf norm ≤ 0.001176470588235294: 74.0 %
  #Linf norm ≤ 0.00392156862745098: 67.4 %
  #Linf norm ≤ 0.01568627450980392: 60.9 %
  
"FGSM: ViT + Resnet(CNN)"   model = timm.create_model('vit_base_r50_s16_224_in21k', pretrained=True).eval().to(device) 
clean accuracy:  87.1 %
robust accuracy for perturbations with
  Linf norm ≤ 0     : 87.1 %
  Linf norm ≤ 0.0003921568627450981: 76.6 %
  Linf norm ≤ 0.001176470588235294: 62.5 %
  Linf norm ≤ 0.00392156862745098: 46.9 %
  Linf norm ≤ 0.01568627450980392: 35.3 %
#%%  
"PGD: ResNet"
 # Linf norm ≤ 0     : 68.4 %
  #Linf norm ≤ 0.0003921568627450981: 51.0 %
  #Linf norm ≤ 0.001176470588235294: 24.1 %
  #Linf norm ≤ 0.00392156862745098:  0.4 %
  #Linf norm ≤ 0.01568627450980392:  0.0 %
"PGD: EfficientNet"
#  Linf norm ≤ 0     : 72.2 %
#  Linf norm ≤ 0.0003921568627450981: 61.3 %     ViT > EfficientNet
#  Linf norm ≤ 0.001176470588235294: 43.8 %      ViT > EfficientNet
#  Linf norm ≤ 0.00392156862745098: 12.7 %       EfficientNet > ViT
#  Linf norm ≤ 0.011764705882352941:  0.6 %      EfficientNet > ViT

"PGD: Vit"  
#robust accuracy for perturbations with
 # Linf norm ≤ 0     : 88.6 %
  #Linf norm ≤ 0.0003921568627450981: 78.1 %    ViT > EfficientNet
  #Linf norm ≤ 0.001176470588235294: 56.7 %     ViT > EfficientNet
  #Linf norm ≤ 0.00392156862745098: 12.5 %      EfficientNet > ViT
  #Linf norm ≤ 0.01568627450980392:  0.0 %      EfficientNet > ViT

"PGD: DeiT"
##  Linf norm ≤ 0     : 79.8 %
  #Linf norm ≤ 0.0003921568627450981: 64.9 %
  #Linf norm ≤ 0.001176470588235294: 37.7 %
  #Linf norm ≤ 0.00392156862745098:  6.7 %
  #Linf norm ≤ 0.011764705882352941:  0.2 %
  
"PGD: Swin"  "Not robust to PGD attacks"          # 800
#clean accuracy:  87.6 %
#robust accuracy for perturbations with
 # Linf norm ≤ 0     : 87.6 %
  #Linf norm ≤ 0.0003921568627450981: 60.0 %
 # Linf norm ≤ 0.001176470588235294: 23.5 %
  #Linf norm ≤ 0.00392156862745098:  1.1 %
  #Linf norm ≤ 0.01568627450980392:  0.0 %
#%%
"Fool_attack: resnet"
#Linf norm ≤ 0     : 68.4 %
#  Linf norm ≤ 0.0003921568627450981: 48.4 %
 # Linf norm ≤ 0.001176470588235294: 22.2 %
  #Linf norm ≤ 0.00392156862745098:  0.4 %
  #Linf norm ≤ 0.01568627450980392:  0.0 %

"Fool_attack: Efficient"
#robust accuracy for perturbations with
 # Linf norm ≤ 0     : 80.4 %
  #Linf norm ≤ 0.0003921568627450981: 69.2 %
  #Linf norm ≤ 0.001176470588235294: 46.6 %
  #Linf norm ≤ 0.00392156862745098: 15.3 %
  #Linf norm ≤ 0.01568627450980392:  1.2 %
  
"Fool_attack: ViT"
#  Linf norm ≤ 0     : 88.6 %
 # Linf norm ≤ 0.0003921568627450981: 77.7 %
  #Linf norm ≤ 0.001176470588235294: 59.1 %
  #Linf norm ≤ 0.00392156862745098: 22.9 %
  #Linf norm ≤ 0.01568627450980392:  0.4 %
  
"Fool_attack: DeiT"
#robust accuracy for perturbations with
 # Linf norm ≤ 0     : 86.6 %
 # Linf norm ≤ 0.0003921568627450981: 64.6 %
 # Linf norm ≤ 0.001176470588235294: 24.1 %
#  Linf norm ≤ 0.00392156862745098:  1.0 %
 # Linf norm ≤ 0.01568627450980392:  0.0 %

"Fool_attack: Swin"
#Linf norm ≤ 0     : 87.6 %
 # Linf norm ≤ 0.0003921568627450981: 42.4 %
  #Linf norm ≤ 0.001176470588235294:  5.3 %
#  Linf norm ≤ 0.00392156862745098:  0.0 %
  #Linf norm ≤ 0.01568627450980392:  0.0 %

"ViT > Efficient > DeiT > ResNet > Swin"


#%%
"Test data: 800 test dataset, 5 Models tested "
# Adversarial attack: maximizing the inner optimization problem
"FGSM:     Swin > DeiT > ViT_Res > ViT > EfficientNet > ResNet"   - Larger epsilon(1/255, 4/255)
"PGD:      EfficientNet > ViT > DeiT > Swin > ResNet"
"DeepFool: ViT > Efficient > DeiT > ResNet > Swin"
# minimal perturbation to fool
# https://velog.io/@wilko97/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-DeepFool-a-simple-and-accurate-method-to-fool-deep-neural-networks-CVPR-2016

"1. ViT models are robust on FGSM attacks but not on PGD(stronger attack) compared to EfficientNet"
#   PGD: Multiple steps with learning rate
#  -> Multiple steps of Gradient attacks affects Robustness of Transformer?
#  ex. exclude shifted layer??
  

#  Shifted windows -> Improve clean accuracy but hurt adversarial robustness
"possible reason: shifted windows limiting self-attention computation harms Robustness"

"3. Feautre learned by ViT would be more generalizable and robust to perturabtion of FGSM, DeepFool not PGD" 

#Future work: 
#1. Look at frequency analysis of features -> How does learning low-level features affects robustness? 
#   -> Literature or library? 
"2. How model size affects Clean Accuracy and Robustness?  ex. CNNs influences compare them"
#3. ViT + ResNet(CNNs) : Combining them
#  How Convolution layer affects ViT robustness? 
#  or Increasing the Transformer Blocks improve robustness? 

"FGSM:     Swin > DeiT > ViT > EfficientNet > ResNet"  
"PGD:      EfficientNet > ViT > DeiT > Swin > ResNet"
"DeepFool: ViT > Efficient > DeiT > ResNet > Swin"

"PGD, DeepFool: Swin(shifted window), DeiT(efficient) is less robust than ViTs, EfficientNet"
#-> shifted windows which limits self-attention computation could harm Robustness?
#-> Distillation token could harm Robustness? 

#%%
"Q1. Parameters of PGD - How do steps, step_size affects robustness?"  # AutoATttack
# Steps are hyperparameter to experiment but how do we fix (step_size and epsilon) in this case?  
#Since the step size α is on the same scale as the total perturbation bound ϵ
# it makes sense to choose α to be some reasonably small fraction of ϵ, 
# and then choose the number of iterations to be a small multiple of ϵ/α => ex. 4, 8, 10 multiply

"Origianl-PGD: Vit"  
#robust accuracy for perturbations with
 # Linf norm ≤ 0     : 88.6 %
  #Linf norm ≤ 0.0003921568627450981: 78.1 %    ViT > EfficientNet
  #Linf norm ≤ 0.001176470588235294: 56.7 %     ViT > EfficientNet
  #Linf norm ≤ 0.00392156862745098: 12.5 %      EfficientNet > ViT
  #Linf norm ≤ 0.01568627450980392:  0.0 %      EfficientNet > ViT
1. step_size = eps/4
robust accuracy for perturbations with
  Linf norm ≤ 0     : 88.6 %
  Linf norm ≤ 0.0003921568627450981: 76.2 %
  Linf norm ≤ 0.001176470588235294: 46.2 %
  Linf norm ≤ 0.00196078431372549: 25.4 %
  Linf norm ≤ 0.0031372549019607846:  6.6 %
  Linf norm ≤ 0.00392156862745098:  3.2 %
  Linf norm ≤ 0.01568627450980392:  0.0 %
2. step_size=  eps/8 
robust accuracy for perturbations with
  Linf norm ≤ 0     : 88.6 %
  Linf norm ≤ 0.0003921568627450981: 76.0 %
  Linf norm ≤ 0.001176470588235294: 46.7 %
  Linf norm ≤ 0.00196078431372549: 25.7 %
Linf norm ≤ 0.0031372549019607846:  7.7 %
  Linf norm ≤ 0.00392156862745098:  4.0 %
  Linf norm ≤ 0.01568627450980392:  0.0 %
3. step_size = eps/12
robust accuracy for perturbations with
  Linf norm ≤ 0     : 88.6 %
  Linf norm ≤ 0.0003921568627450981: 75.9 %
  Linf norm ≤ 0.001176470588235294: 48.0 %
  Linf norm ≤ 0.00392156862745098:  4.7 %
  Linf norm ≤ 0.01568627450980392:  0.0 %
  
###############################################################################
"PGD: EfficientNet"
#  Linf norm ≤ 0     : 72.2 %
#  Linf norm ≤ 0.0003921568627450981: 61.3 %     ViT > EfficientNet
#  Linf norm ≤ 0.001176470588235294: 43.8 %      ViT > EfficientNet
#  Linf norm ≤ 0.00392156862745098: 12.7 %       EfficientNet > ViT
#  Linf norm ≤ 0.011764705882352941:  0.6 %      EfficientNet > ViT

1. step_size = eps/4
robust accuracy for perturbations with
  Linf norm ≤ 0     : 80.4 %
  Linf norm ≤ 0.0003921568627450981: 70.2 %
  Linf norm ≤ 0.001176470588235294: 46.0 %
  Linf norm ≤ 0.00196078431372549:  28.4 %
  Linf norm ≤ 0.0031372549019607846: 13.6 %
  Linf norm ≤ 0.00392156862745098:  9.7 %
  Linf norm ≤ 0.01568627450980392:  0.0 %
2. step_size=  eps/8 
robust accuracy for perturbations with
  Linf norm ≤ 0     : 80.4 %
  Linf norm ≤ 0.0003921568627450981: 70.3 %
  Linf norm ≤ 0.001176470588235294: 46.6 %
  Linf norm ≤ 0.00196078431372549: 30.3 %
Linf norm ≤ 0.0031372549019607846: 15.3 %
  Linf norm ≤ 0.00392156862745098: 10.8 %
  Linf norm ≤ 0.01568627450980392:  0.0 %
3. step_size = eps/12  
  Linf norm ≤ 0     : 80.4 %
  Linf norm ≤ 0.0003921568627450981: 70.3 %
  Linf norm ≤ 0.001176470588235294: 47.1 %
  Linf norm ≤ 0.00196078431372549: 31.5 %
  Linf norm ≤ 0.0031372549019607846: 17.3 %
  Linf norm ≤ 0.00392156862745098: 12.5 %
  Linf norm ≤ 0.01568627450980392:  0.0 %
##################################################
"PGD: ViT_Hybrid"



"Q1. Parameters of PGD - How do steps, step_size affects robustness?" 
"PGD Insight(3 different step_size): Vit is more robust than EfficientNet is more robust than ViT when epsilon is smaller than 0.5/255 "
step_size changes depending on epsilons(each epsilon has different step_size) "eps/4, eps/8, eps/12" - 
step_size (learning rate)??
1) small : Finding optimum takes time 
2) Huge  : may skip optimum

"Q2. Size of epsilons could affect PGD, ViTs? - step_size will be changed depending on epsilons"
Original epsilons = [0, 0.1/255, 0.3/255, 1/255, 4/255]  
Try smaller epsilons -> [0.5/255, 0.8/255]
# Accuracy   ViT > Efficient 
# 0.1/255    ViT > EfficientNet
# 0.3/255    ViT > EfficientNet
  0.5/255    EfficientNet > ViT
  0.8/255    EfficientNet > ViT
# 1/255      EfficientNet > ViT
# 4/255      EfficientNet > ViT



#%%
Question 1: ViT has higher clean accuracy compared to CNNs. In this case, can we say ViT is more robust with smaller epsilons of pertubation?
  "Accuracy   ViT > Efficient"   88.6 > 80.4
  0.1/255    ViT > EfficientNet
  0.3/255    ViT > EfficientNet
  0.5/255    EfficientNet > ViT
  0.8/255    EfficientNet > ViT

"Q3. How to analyze the result of Deepfool?" - Also step_size????

"Q5.ex. CNN - Purtubated images(around 200) - Overshoot in the edges like rings - How about ViT ??? "
- Vit_explain(Attention Visualization) - PGD (CNNs vs ViT) 이해하기  

"Q4. ViT + ResNet(CNNs) - Robustness " - How Convolution layer affects ViT robustness? 
  
FGSM: ViT + Resnet(CNN): model = timm.create_model('vit_base_r50_s16_224_in21k', pretrained=True).eval().to(device) 
Accuracy of model is not correct!!!!

clean accuracy:  60.5 % XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
robust accuracy for perturbations with
  Linf norm ≤ 0     : 60.5 %
  Linf norm ≤ 0.0003921568627450981: 43.7 %
  Linf norm ≤ 0.001176470588235294: 21.9 %
  Linf norm ≤ 0.00392156862745098:  3.9 %
  Linf norm ≤ 0.01568627450980392:  0.4 %
  
  
"Q5. Increasing the Transformer Blocks improve robustness? (model size)? 
: Compare results of ViT and CNNs

#=================================================
"New_Q. Evaluate confidence of Rob"
#- ex. 0.5- dog, 0.3 - cat   --> CNN is confident 99%  
#PGD: ex. confident on wrong prediction. 