# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:14:40 2022

@author: ChangGun Choi
"""

#%%
# https://colab.research.google.com/drive/1muZ4QFgVfwALgqmrfOkp7trAvqDemckO?usp=sharing
"Test data: 819 dataset"

"FGSM : Resnet_18"
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
  
"FGSM: ResNet_50"
clean accuracy:  84.2 %
robust accuracy for perturbations with
  Linf norm ≤ 0     : 84.2 %
  Linf norm ≤ 0.0003921568627450981: 69.9 %
  Linf norm ≤ 0.001176470588235294: 53.9 %
  Linf norm ≤ 0.00392156862745098: 35.0 %
  Linf norm ≤ 0.01568627450980392: 24.9 %
  
  
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
#clean accuracy:  87.6 %
#robust accuracy for perturbations with
 # Linf norm ≤ 0     : 87.6 %
  #Linf norm ≤ 0.0003921568627450981: 76.6 %
  #Linf norm ≤ 0.001176470588235294: 74.0 %
  #Linf norm ≤ 0.00392156862745098: 67.4 %
  #Linf norm ≤ 0.01568627450980392:  60.9 %
  
"FGSM: ViT + Resnet(CNN)"   model = timm.create_model('vit_base_r50_s16_224', pretrained=True).eval().to(device) 
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
  #Linf norm ≤ 0.001176470588235294: 24.1 %   [68.4, 51, 24, 0.4, 0]
  #Linf norm ≤ 0.00392156862745098:  0.4 %
  #Linf norm ≤ 0.01568627450980392:  0.0 %
  
"MobileNet"
clean accuracy:  78.1 %
robust accuracy for perturbations with
  Linf norm ≤ 0     : 78.1 %                     [68.4, 51, 24, 0.4, 0], [84.2, 68.8,42.59.2, 0], [78.1, 61.5,34.3, 5.8, 0 ]
  Linf norm ≤ 0.0003921568627450981: 61.5 %
  Linf norm ≤ 0.001176470588235294: 34.3 %
  Linf norm ≤ 0.00392156862745098:  5.8 %
  Linf norm ≤ 0.01568627450980392:  0.0 %  
  
"PGD: VGG"
clean accuracy:  73.0 %
robust accuracy for perturbations with
  Linf norm ≤ 0     : 73.0 %
  Linf norm ≤ 0.0003921568627450981: 58.5 %    
  Linf norm ≤ 0.001176470588235294: 30.0 %
  Linf norm ≤ 0.00392156862745098:  1.7 %
  Linf norm ≤ 0.01568627450980392:  0.1 %
    
"PGD: EfficientNet"
  Linf norm ≤ 0     : 80.4 %
  Linf norm ≤ 0.0003921568627450981: 71.6 %   
  Linf norm ≤ 0.001176470588235294: 52.5 %
  Linf norm ≤ 0.00392156862745098: 20.6 %
  Linf norm ≤ 0.01568627450980392:  0.1 %
  
  EfficientB4??
robust accuracy for perturbations with
  Linf norm ≤ 0     : 74.5 %
  Linf norm ≤ 0.0003921568627450981: 61.3 %
  Linf norm ≤ 0.001176470588235294: 37.9 %
  Linf norm ≤ 0.00392156862745098:  6.0 %
  Linf norm ≤ 0.01568627450980392:  0.0 %

"PGD: ResNet_50" 
clean accuracy:  84.2 %
robust accuracy for perturbations with
  Linf norm ≤ 0     : 84.2 %
  Linf norm ≤ 0.0003921568627450981: 68.8 %     [68.4, 51, 24, 0.4, 0], [84.2, 68.8,42.59.2, 0]
  Linf norm ≤ 0.001176470588235294: 42.5 %     
  Linf norm ≤ 0.00392156862745098:  9.2 %
  Linf norm ≤ 0.01568627450980392:  0.0 %

"PGD: Vit"           [88.6,  78.1,  56.7,  12.5, 0]
#robust accuracy for perturbations with  [
 # Linf norm ≤ 0     : 88.6 %
  #Linf norm ≤ 0.0003921568627450981: 78.1 %    ViT > EfficientNet
  #Linf norm ≤ 0.001176470588235294: 56.7 %     ViT > EfficientNet
  #Linf norm ≤ 0.00392156862745098: 12.5 %      EfficientNet > ViT
  #Linf norm ≤ 0.01568627450980392:  0.0 %      EfficientNet > ViT

"PGD: DeiT" 다시다시다
 Linf norm ≤ 0     : 86.6 %
  Linf norm ≤ 0.0003921568627450981: 71.6 %
  Linf norm ≤ 0.001176470588235294: 45.1 %
  Linf norm ≤ 0.00392156862745098:  8.3 %
  Linf norm ≤ 0.01568627450980392:  0.1 %
  
"PGD: Swin"  "Not robust to PGD attacks"          # 800
#clean accuracy:  87.6 %
#robust accuracy for perturbations with
 # Linf norm ≤ 0     : 87.6 %
  #Linf norm ≤ 0.0003921568627450981: 60.0 %
 # Linf norm ≤ 0.001176470588235294: 23.5 %
  #Linf norm ≤ 0.00392156862745098:  1.1 %
  #Linf norm ≤ 0.01568627450980392:  0.0 %
"PGD: ViT_Res"  "Not robust to PGD attacks"  
  Linf norm ≤ 0     : 87.1 %
  Linf norm ≤ 0.0003921568627450981: 72.2 %
  Linf norm ≤ 0.001176470588235294: 39.1 %
  Linf norm ≤ 0.00392156862745098:  2.8 %
  Linf norm ≤ 0.01568627450980392:  0.0 %
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
"Accuracy" : Vit >  Swin > Hybrid > DeiT > Resnet50> efficient > VGG> ResNet
FGSM:       Swin > DeiT > "Hybrid" > ViT > "ResNet_50" > EfficientNet > ResNet  
PGD:      EfficientNet > ViT > ResNet_50 > DeiT > Swin > ResNet"
DeepFool: ViT > Efficient > DeiT > ResNet > Swin"
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
# and then choose the number of iterations to be a small multiple of ϵ/α => ex. 4, 8, 12 multiply


###############################################################################
"PGD: EfficientNet"

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
##############################################################

"Origianl-PGD: Vit"  

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
  Linf norm ≤ 0.00196078431372549: 27.2 %
  Linf norm ≤ 0.0031372549019607846:  8.1 %
  Linf norm ≤ 0.00392156862745098:  4.7 %
  Linf norm ≤ 0.01568627450980392:  0.0 %
  ######################################################
  
"PGD: ResNet50"

  Linf norm ≤ 0     : 84.2 %
  Linf norm ≤ 0.0003921568627450981: 65.1 %
  Linf norm ≤ 0.001176470588235294: 28.9 %
  Linf norm ≤ 0.00196078431372549: 12.5 %
  Linf norm ≤ 0.0031372549019607846:  4.0 %
  Linf norm ≤ 0.00392156862745098:  2.6 %
                                      0
  Linf norm ≤ 0     : 84.2 %
Linf norm ≤ 0.0003921568627450981: 65.1 %
Linf norm ≤ 0.001176470588235294: 30.4 %
Linf norm ≤ 0.00196078431372549: 14.0 %
Linf norm ≤ 0.0031372549019607846:  4.9 %
Linf norm ≤ 0.00392156862745098:  2.9 %

Linf norm ≤ 0     : 84.2 %
Linf norm ≤ 0.0003921568627450981: 65.1 %
Linf norm ≤ 0.001176470588235294: 31.9 %
Linf norm ≤ 0.00196078431372549: 15.6 %
Linf norm ≤ 0.0031372549019607846:  5.9 %
Linf norm ≤ 0.00392156862745098:  3.4 %
##################################################
"PGD: ViT_Hybrid"
1. step_size = eps/4
  Linf norm ≤ 0     : 87.1 %
  Linf norm ≤ 0.0003921568627450981: 72.2 %
  Linf norm ≤ 0.001176470588235294: 39.1 %
  Linf norm ≤ 0.00196078431372549: 18.5 %
 Linf norm ≤ 0.0031372549019607846:  5.1 %
  Linf norm ≤ 0.00392156862745098:  2.8 %
  Linf norm ≤ 0.01568627450980392:  0.0 %
2. step_size=  eps/8 
robust accuracy for perturbations with
  Linf norm ≤ 0     : 87.1 %
  Linf norm ≤ 0.0003921568627450981: 72.3 %
  Linf norm ≤ 0.001176470588235294: 40.4 %
  Linf norm ≤ 0.00196078431372549: 19.9 %
Linf norm ≤ 0.0031372549019607846:  6.1 %
  Linf norm ≤ 0.00392156862745098:  3.1 %
  Linf norm ≤ 0.01568627450980392:  0.0 %
3. step_size = eps/12
  Linf norm ≤ 0     : 87.1 %
  Linf norm ≤ 0.0003921568627450981: 72.2 %
  Linf norm ≤ 0.001176470588235294: 41.8 %
  Linf norm ≤ 0.00196078431372549: 21.9 %
  Linf norm ≤ 0.0031372549019607846:  7.5 %
  Linf norm ≤ 0.00392156862745098:  3.7 %
  Linf norm ≤ 0.01568627450980392:  0.0 %
  


#%%
"Q1. Parameters of PGD - How does step_size affects robustness?" 
"PGD Insight(3 different step_size): Vit is more robust than EfficientNet when epsilon is smaller than 0.5/255 "
step_size changes depending on epsilons(each epsilon has different step_size) "eps/4, eps/8, eps/12" - 
step_size (learning rate)??
1) small : Finding optimum takes time 
2) Huge  : may skip optimum

"Q1-2. Size of epsilons could affect PGD, ViTs? - step_size will be changed depending on epsilons"
Original epsilons = [0, 0.1/255, 0.3/255, 1/255, 4/255]  
Try smaller epsilons to verify the PGD results -> [0.5/255, 0.8/255]
#######################################################################################################
"Test data: 800 dataset, 7 Models tested 
- More experiments with Hybrid and ResNet50 models. 

Accuracy   : Vit >  Swin > "Hybrid" > DeiT > "CNN" (Resnet50 > EfficientNet > ResNet)
FGSM:       Swin >  DeiT > "Hybrid" > ViT  > "CNN" (ResNet50> EfficientNet > ResNet)
PGD:
  0.1/255    ViT > "Hybrid" > CNN (EfficientNet > ResNet50 > Resnet)
  0.3/255    ViT > "Hybrid" > CNN (EfficientNet > ResNet50 > Resnet)
  0.5/255    "EfficientNet" >  ViT > "Hybrid" > "ResNet50" > Resnet
  0.8/255    "EfficientNet" >  ViT > "Hybrid" > "ResNet50" > Resnet
  1/255      "EfficientNet" >  ViT > "Hybrid" > "ResNet50" > Resnet
  4/255      "EfficientNet" >  ViT > "Hybrid" > "ResNet50" > Resnet

"Analysis of Results"
0. ViT more robust than general CNNs
: e ViTs learn less high-frequency features
 One possible explanation is that the introduced modules improve the classification accuracy 
by remembering the low-level structures that repeatedly appear in the training dataset. 
These structures such as edges and lines are high-frequent and sensitive to perturbations. 
Learning such featuresmakes the model more vulnerable to adversarial attacks

1. PGD: ViT is more robust than Hybrid and general CNNs except "EfficientNet"
"Q. Robustness of Hybrid " - How Convolution layer affects ViT robustness?
-> Results of "Hybrid" and CNNs_"ResNet 50" shows that Convolutional layer have bad affect on Robustness 
-> However, "EfficientNet" is strong to PGD (My finding)

"Q. Why EfficientNet is Robust to PGD? "
Reason could be Training or Architecture?
- Training - Learning rate? 

2. Results of CNNs comparison
Accuracy and FGSM:  Resnet50 > EfficientNet  (Higher accuracy, More Robust)
PGD              :  EfficientNet > Resnet50

"Q CNN - Purtubated images(around 200) - Overshoot in the edges like rings"
   ViT - How does Purtubated images look like??
   
- "Q. Visualization of CNNs??" - GradCam ++
ex. CNNs - Strided COnvloution - attacks higlyfrequency  - attacking Pooling operation - changes globally 

- Vit_explain(Attention Visualization) - PGD (CNNs vs ViT)   
2. Origianl RObustness??

"Q6. Evaluate confidence of Robustness"
- ex. 0.5- dog, 0.3 - cat   --> CNN is confidence score 99%  
PGD: ex. confident on wrong prediction. 

  
"Q5. Increasing the Transformer Blocks improve robustness? (model size)? 
: Compare results of ViT and CNNs

#===============================================
#%%
Calibrated ECE
1. "ViT has higher ECE"

1) ResNet
ECE = 0.01%_0_resnet50
MCE = 33.28%_0_resnet50

ECE = 0.04%_0.1_resnet50
MCE = 41.19%_0.1_resnet50

ECE = 0.10%_0.3_resnet50
MCE = 56.64%_0.3_resnet50

ECE = 0.14%_1.0_resnet50
MCE = 93.03%_1.0_resnet50

ECE = 0.11%_4.0_resnet50
MCE = 98.32%_4.

2) VGG
ECE = 0.01%_0_VGG
MCE = 9.30%_0_VGG    

ECE = 0.02%_0.1_VGG
MCE = 22.94%_0.1_VGG

ECE = 0.08%_0.3_VGG
MCE = 60.88%_0.3_VGG

ECE = 0.18%_1.0_VGG
MCE = 98.58%_1.0_VGG

ECE = 0.14%_4.0_VGG
MCE = 99.53%_4.0_VGG

3) Efficient
ECE = 0.04%_0_efficient
MCE = 24.68%_0_efficient

ECE = 0.03%_0.1_efficient
MCE = 24.58%_0.1_efficient

ECE = 0.03%_0.3_efficient
MCE = 87.78%_0.3_efficient

ECE = 0.10%_1.0_efficient
MCE = 96.55%_1.0_efficient

ECE = 0.19%_4.0_efficient
MCE = 99.58%_4.0_efficient

4)Vit
ECE = 0.00%_0_vit
MCE = 9.88%_0_vit

ECE = 0.01%_0.1_vit
MCE = 9.55%_0.1_vit

ECE = 0.05%_0.3_vit
MCE = 33.70%_0.3_vit

ECE = 0.14%_1.0_vit
MCE = 93.17%_1.0_vit

ECE = 0.20%_4.0_vit
MCE = 99.19%_4.0_vit

5) ViT-Hybrid
    
ECE = 0.00%_0_vit_hybrid
MCE = 10.99%_0_vit_hybrid

ECE = 0.02%_0.1_vit_hybrid
MCE = 25.32%_0.1_vit_hybrid

ECE = 0.06%_0.3_vit_hybrid
MCE = 39.88%_0.3_vit_hybrid

ECE = 0.15%_1.0_vit_hybrid
MCE = 94.07%_1.0_vit_hybrid

ECE = 0.20%_4.0_vit_hybrid
MCE = 99.29%_4.0_vit_hybrid

#%%
from matplotlib import pyplot as plt
plt.plot(xPoints, [0, 0.03, 0.08, 0.18])  # resnet 18
plt.plot([0, 0.1, 0.3, 1], [0.01, 0.04, 0.10, 0.14])
plt.plot([0, 0.1, 0.3, 1], [0.01, 0.02, 0.08, 0.18])
plt.plot(xPoints, [0.02, 0.02, 0.04, 0.13])   # mobile
plt.plot([0, 0.1, 0.3, 1], [0.04, 0.03, 0.03, 0.10])
plt.plot([0, 0.1, 0.3, 1], [ 0  ,  0.01,0.05, 0.14])

plt.plot([0, 0.1, 0.3, 1], [0.1, 0.09, 0.04, 0.08])
plt.plot([0, 0.1, 0.3, 1],  [0.11, 0.07, 0.06, 0.15])
plt.plot([0, 0.1, 0.3, 1], [0.00, 0.02, 0.06, 0.15])

plt.xlabel('Epsilons')
plt.ylabel('ECE')
plt.title('Expected Calibration Error')
plt.legend(['resnet 18', 'ResNet50', 'VGG', ' mobile', 'Efficient', 'ViT', 'DeiT', 'Swin','ViT-Hybrid'])
plt.show()

from matplotlib import pyplot as plt
#fig, (ax1, ax2) = plt.subplots(2,1)
#fig.suptitle('Aligning x-axis using sharex')


plt.plot(xPoints, [0, 0.03, 0.08, 0.18])
plt.plot(xPoints, [0.01, 0.02, 0.08, 0.18])
plt.plot(xPoints, [0.02, 0.02, 0.04, 0.13])
plt.plot(xPoints, [0.04, 0.03, 0.03, 0.10])

#ax1.subplot(1, 2, 1) # row 1, col 2 index 1
plt.xlabel('Epsilons')
plt.ylabel('ECE')
plt.title('Expected Calibration Error')
plt.legend(['ResNet18', 'VGG', 'mobilenet','Efficient'])
plt.show()

#ax2.subplot(1, 2, 2) # index 2
xPoints = [0, 0.1, 0.3, 1]
plt.plot(xPoints, [68.4, 51, 24, 0.4])
plt.plot(xPoints, [73.0, 58.5, 30, 1.7])  
plt.plot(xPoints, [78.1, 61.5,34.3, 5.8] )
plt.plot(xPoints, [80.4, 71.6,52.5,20.6])


plt.xlabel('Epsilons')
plt.ylabel('Robustness')
plt.title('Robust Accuracy')
plt.legend(['ResNet18', 'VGG', 'mobilenet','Efficient'])
plt.show()

#%% Resnet18, VGG, Efficient, ViT
xPoints = [0, 0.1, 0.3, 1]
plt.plot(xPoints, [0, 0.03, 0.08, 0.18])
plt.plot(xPoints, [0.01, 0.02, 0.08, 0.18])
plt.plot(xPoints, [0.04, 0.03, 0.03, 0.10])
plt.plot(xPoints, [ 0  ,  0.01,0.05, 0.14])
#ax1.subplot(1, 2, 1) # row 1, col 2 index 1
plt.xlabel('Epsilons')
plt.ylabel('ECE')
plt.title('Expected Calibration Error')
plt.legend(['ResNet18', 'VGG','Efficient', 'ViT'])
plt.show()

plt.plot(xPoints, [68.4, 51, 24, 0.4])
plt.plot(xPoints, [73.0, 58.5, 30, 1.7])  
plt.plot(xPoints, [80.4, 71.6,52.5,20.6])
plt.plot(xPoints, [88.6,  78.1,  56.7,  12.5])

plt.xlabel('Epsilons')
plt.ylabel('Robustness')
plt.title('Robust Accuracy')
plt.legend(['ResNet18', 'VGG','Efficient', 'ViT'])
plt.show()




# plt.plot(xPoints, [84.2, 68.8, 42.5, 9.2, 0]) ResNet50 Robustness
xPoints = [0, 0.1, 0.3, 1]
from matplotlib import pyplot as plt
plt.plot(xPoints, [ 0  ,  0.01,0.05, 0.14])
plt.plot(xPoints,  [0.00, 0.02, 0.06, 0.15])
plt.plot(xPoints,  [0.1, 0.09, 0.04, 0.08])
plt.plot(xPoints,  [0.11, 0.07, 0.06, 0.15])

plt.xlabel('Epsilons')
plt.ylabel('ECE')
plt.title('Expected Calibration Error')
plt.legend(['ViT','ViT-Hybrid', 'DeiT', 'Swin'])
plt.show()

plt.plot(xPoints, [88.6,  78.1,  56.7,  12.5, 0])

plt.xlabel('Epsilons')
plt.ylabel('Robustness')
plt.title('Robust Accuracy')
plt.legend(['ViT','ViT-Hybrid', 'DeiT', 'Swin'])
plt.show()

#%%
#plt.plot([0, 0.1, 0.3, 1, 4], [0.01, 0.03, 0.08, 0.17, 0.18]): 'Efficient_B4',

#%%
ViT

"High"
  Linf norm ≤ 0     : 88.6 %
  Linf norm ≤ 0.0003921568627450981: 83.6 %
  Linf norm ≤ 0.001176470588235294: 74.9 %
  Linf norm ≤ 0.00392156862745098: 50.5 %
  Linf norm ≤ 0.01568627450980392: 27.3 %

"Low"
 Linf norm ≤ 0     : 88.6 %
  Linf norm ≤ 0.0003921568627450981: 87.6 %
  Linf norm ≤ 0.001176470588235294: 85.4 %
  Linf norm ≤ 0.00392156862745098: 79.8 %
  Linf norm ≤ 0.01568627450980392: 72.8 %

Efficient : Weak at high 
"High"
  Linf norm ≤ 0     : 80.4 %
  Linf norm ≤ 0.0003921568627450981: 75.6 %
  Linf norm ≤ 0.001176470588235294: 65.2 %
  Linf norm ≤ 0.00392156862745098: 34.9 %
  Linf norm ≤ 0.01568627450980392:  4.5 %
  
"Low"

  Linf norm ≤ 0     : 80.4 %
  Linf norm ≤ 0.0003921568627450981: 79.4 %
  Linf norm ≤ 0.001176470588235294: 78.4 %
  Linf norm ≤ 0.00392156862745098: 75.6 %
  Linf norm ≤ 0.01568627450980392: 72.9 %
  
ViT_hybrid 
"low"
robust accuracy for perturbations with
  Linf norm ≤ 0     : 87.1 %
  Linf norm ≤ 0.0003921568627450981: 85.4 %
  Linf norm ≤ 0.001176470588235294: 83.8 %
  Linf norm ≤ 0.00392156862745098: 80.8 %
  Linf norm ≤ 0.01568627450980392: 78.7 %
  
  


