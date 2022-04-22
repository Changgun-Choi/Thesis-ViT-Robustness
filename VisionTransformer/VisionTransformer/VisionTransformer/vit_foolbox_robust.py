#Server
# git stash
# cd "/home/cchoi/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer"
#conda activate thesis
# python vit_foolbox_robust.py --model_name vit --attack_name FGSM --batch_size 16 --data_divide 10 --data_path server

#!/usr/bin/env python3
#cd "C:/Users/ChangGun Choi/Team Project/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer"
#python vit_foolbox_robust.py --model_name resnet --attack_name FGSM --batch_size 8 --data_divide 100 --data_path server
# nvidia-smi
"""
A simple example that demonstrates how to run a single attack against
a PyTorch ResNet-18 model for different epsilons and how to then report
the robust accuracy.
"""
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD
import foolbox.attacks as attack
import foolbox.attacks as fa
import foolbox as fb
import argparse 
import timm
#assert timm.__version__ == "0.3.2"
import gc
gc.collect()
torch.cuda.empty_cache()
import matplotlib.pyplot as plt
from functools import partial
from functools import partial as pll
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vision Transformer')
    parser.add_argument('--model_name', default='vit', type=str, help='data name')    
    parser.add_argument('--attack_name', default='LinfPGD', type=str, help='attack name') 
    parser.add_argument('--batch_size',default = 8, type = int)
    parser.add_argument('--data_divide',default = 100, type = int, help = 'multiply by 0.01')  # /100 : 500  args.data_path 
    parser.add_argument('--data_path',default = 'local', type = str) 
    args = parser.parse_args()  
    #print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   ############################################
    #device = "cpu"
    
    if args.model_name == 'resnet':
        model = torchvision.models.resnet18(pretrained=True).eval().to(device)
        
    elif args.model_name == 'efficient':  
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True).eval().to(device)
    
    elif args.model_name == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True).eval().to(device)    
        
    elif args.model_name == 'deit': 
        #model = torch.hub.load('facebookresearch/deit:main','deit_base_patch16_224', pretrained=True).eval().to(device)
        def deit_base_patch16_224(pretrained=False, **kwargs):
            model = VisionTransformer(
                patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
            model.default_cfg = _cfg()
            if pretrained:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                    map_location="cpu", check_hash=True
                )
                model.load_state_dict(checkpoint["model"])
            return model
        model = deit_base_patch16_224(pretrained=True).eval().to(device)  
   
    elif args.model_name == 'swin_base':
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True).eval().to(device)
        
    elif args.model_name == 'swin_path4':
        model = timm.create_model('swin_s3_base_224', pretrained=True).eval().to(device)

        model = timm.create_model('swin_large_patch4_window7_224', pretrained=True)

        
        model = timm.create_model('swin_large_patch4_window12_384_in22k', in_chans = 3, pretrained = True,)
        model = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=True)
        from transformers import SwinModel
        import timm
        model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        from timm.models import swin_base_patch4_window7_224_in22k
        timm.list_models(pretrained=True) 
        timm.list_models('*sw*')
   
        
        
    elif args.model_name == 'dino_vit':      # clean accuracy:  0.0 %
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=True).eval().to(device)
        
    elif args.model_name == 'resnet50_dino':       # clean accuracy:  0.0 %
        # instantiate a model (could also be a TensorFlow or JAX model)
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50',pretrained=True).eval().to(device) # eval
 
    preprocessing = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-3)   # normalize inside model.  
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),     # Cifar 10 (32 x 32)  / ImageNet 220 x220 
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])    
    epsilons = [0, 0.1/255, 0.3/255, 1/255, 4/255]
    #epsilons = [0, 0.1/255] 
    
    if args.data_path == 'local':
        data_path = 'C:/Users/ChangGun Choi/Team Project/Thesis_data/val'
    elif args.data_path == 'server':
        data_path = '/home/cchoi/Thesis_data/data/val'
        
    testset = torchvision.datasets.ImageNet(data_path, split='val', transform=test_transform)
    sample = list(range(0, len(testset), args.data_divide))   # 16 * 3125 * 0.01 : 500
    valset = torch.utils.data.Subset(testset, sample) 
    val_loader = torch.utils.data.DataLoader(valset, args.batch_size ,drop_last=True)
    
    "clean_accuracy"    
    def get_acc(model, inputs, labels):
        with torch.no_grad():
            predictions = model(inputs).argmax(axis=-1)
            accuracy = (predictions == labels).float().mean()
            return accuracy.item() 
        
    #images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=8))  # samples() has only 20 samples and repeats itself if batchsize > 20
    #clean_acc = accuracy(fmodel, images, labels)
       
    if args.attack_name == 'PGD':  # single adversarial attack (Linf PGD)
        attack = LinfPGD()
        accuracy = 0 
        success = torch.zeros(len(epsilons),args.batch_size).cuda()
        for batch_idx, (image, label) in enumerate(val_loader):
            print("Attack: {}/{}".format(batch_idx+1, len(val_loader)-1))
            images = image.cuda()
            labels = label.cuda()
            #images, labels = ep.astensors(images, labels)
            clean_acc = get_acc(fmodel, images, labels)
            raw_advs, clipped_advs, succ = attack(fmodel, images, labels, epsilons=epsilons)         
            succ = torch.cuda.FloatTensor(succ.detach().cpu().numpy()) # 1) EagerPy -> numpy 2) Numpy -> FloatTensor)
            #print(succ)
            success += succ
            accuracy += clean_acc
            #print(success)
        accuracy = accuracy / len(val_loader)
        print(f"clean accuracy:  {accuracy * 100:.1f} %") 
        success = success/len(val_loader)            #  # succes of Attack (lowering accuracy)
        robust_accuracy = 1 - success.mean(dim = -1) # t.mean(dim=1): Mean of last dimension (different with other dim)
        print("robust accuracy for perturbations with")
        for eps, acc in zip(epsilons, robust_accuracy):
            print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")  
        plt.figure(figsize=(5,5))
        plt.plot(epsilons, robust_accuracy.cpu().numpy()) 
        plt.show()
        
    elif args.attack_name == 'FGSM':
        attack = fa.FGSM()
        accuracy = 0 
        success = torch.zeros(len(epsilons),args.batch_size).cuda()
        for batch_idx, (image, label) in enumerate(val_loader):
            print("Attack: {}/{}".format(batch_idx+1, len(val_loader)-1))
            images = image.cuda()
            labels = label.cuda()
            #images, labels = ep.astensors(images, labels)
            clean_acc = get_acc(fmodel, images, labels)
            raw_advs, clipped_advs, succ = attack(fmodel, images, labels, epsilons=epsilons)         
            succ = torch.cuda.FloatTensor(succ.detach().cpu().numpy()) # 1) EagerPy -> numpy 2) Numpy -> FloatTensor)
            #print(succ)
            success += succ
            accuracy += clean_acc
            #print(success)
        accuracy = accuracy / len(val_loader)
        print(f"clean accuracy:  {accuracy * 100:.1f} %") 
        success = success/len(val_loader)            #  # succes of Attack (lowering accuracy)
        robust_accuracy = 1 - success.mean(dim = -1) # t.mean(dim=1): Mean of last dimension (different with other dim)
        print("robust accuracy for perturbations with")
        #robust_accuracy
        plt.figure(figsize=(5,5))
        plt.plot(epsilons, robust_accuracy.cpu().numpy()) 
        plt.show()
        for eps, acc in zip(epsilons, robust_accuracy):
            print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")    
         
        
    elif args.attack_name == 'DeepFool': 
        attack = fb.attacks.LinfDeepFoolAttack()
        accuracy = 0 
        success = torch.zeros(len(epsilons),args.batch_size).cuda()
        for batch_idx, (image, label) in enumerate(val_loader):
            print("Attack: {}/{}".format(batch_idx+1, len(val_loader)-1))
            images = image.cuda()
            labels = label.cuda()
            #images, labels = ep.astensors(images, labels)
            clean_acc = get_acc(fmodel, images, labels)
            raw_advs, clipped_advs, succ = attack(fmodel, images, labels, epsilons=epsilons)         
            succ = torch.cuda.FloatTensor(succ.detach().cpu().numpy()) # 1) EagerPy -> numpy 2) Numpy -> FloatTensor)
            #print(succ)
            success += succ
            accuracy += clean_acc
            #print(success)
        accuracy = accuracy / len(val_loader)
        print(f"clean accuracy:  {accuracy * 100:.1f} %") 
        success = success/len(val_loader)            #  # succes of Attack (lowering accuracy)
        robust_accuracy = 1 - success.mean(dim = -1) # t.mean(dim=1): Mean of last dimension (different with other dim)
        print("robust accuracy for perturbations with")
        for eps, acc in zip(epsilons, robust_accuracy):
            print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")   
        plt.figure(figsize=(5,5))
        plt.plot(epsilons, robust_accuracy.cpu().numpy()) 
        plt.show()
          
#%%            
    elif args.attack_name == 'multiple_attacks': 
        attacks = [
            fa.FGSM(),   # FGSM= LinfFastGradientAttack(LinfBaseGradientDescent)
            fa.LinfPGD(),
            fa.LinfBasicIterativeAttack(),
            fa.LinfAdditiveUniformNoiseAttack(),
            fa.LinfDeepFoolAttack(),]
        print("epsilons")
        print(epsilons)
        print("")
        
        "attack:List"
        attack_success = np.zeros((len(attacks), len(epsilons), len(images)), dtype=np.bool)
        for i, attack in enumerate(attacks):
            _, _, success = attack(fmodel, images, labels, epsilons=epsilons)
            assert success.shape == (len(epsilons), len(images))
            success_ = success.numpy()
            assert success_.dtype == np.bool
            attack_success[i] = success_
            print(attack)
            print("  ", 1.0 - success_.mean(axis=-1).round(2))
    
        # calculate and report the robust accuracy (the accuracy of the model when
        # it is attacked) using the best attack per sample
        robust_accuracy = 1.0 - attack_success.max(axis=0).mean(axis=-1)
        print("")
        print("-" * 79)
        print("")
        print("worst case (best attack per-sample)")
        print("  ", robust_accuracy.round(2))
        print("")
    
        print("robust accuracy for perturbations with")
        for eps, acc in zip(epsilons, robust_accuracy):
            print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
        plt.plot(epsilons, robust_accuracy.numpy())    
            

#%% "Data_size: 500"

"Multiple: ResNet"       
#clean accuracy:  93.8 %
#epsilons
#[0.0, 0.0002, 0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.003, 0.01, 0.1]

#LinfFastGradientAttack(rel_stepsize=1.0, abs_stepsize=None, steps=1, random_start=False)
#   [0.94 0.88 0.81 0.56 0.44 0.31 0.25 0.06 0.   0.  ]
#LinfProjectedGradientDescentAttack(rel_stepsize=0.03333333333333333, abs_stepsize=None, steps=40, random_start=True)
#   [0.94 0.88 0.81 0.5  0.38 0.12 0.06 0.   0.   0.  ]
#LinfBasicIterativeAttack(rel_stepsize=0.2, abs_stepsize=None, steps=10, random_start=False)
#   [0.94 0.81 0.69 0.38 0.38 0.12 0.06 0.   0.   0.  ]
#LinfAdditiveUniformNoiseAttack()
#   [0.94 0.94 0.94 0.94 0.94 0.94 0.94 0.94 0.94 0.88]
#LinfDeepFoolAttack(steps=50, candidates=10, overshoot=0.02, loss=logits)
#   [0.94 0.81 0.75 0.44 0.31 0.12 0.06 0.   0.   0.  ]

#worst case (best attack per-sample)-
#   [0.94 0.81 0.69 0.38 0.31 0.12 0.06 0.   0.   0.  ]      
#%%
# https://colab.research.google.com/drive/1muZ4QFgVfwALgqmrfOkp7trAvqDemckO?usp=sharing
"Test: 800 data"
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
  
"FGSM: ViT - clean_accuracy: 81.2 % "
#clean accuracy:  88.6 %
#robust accuracy for perturbations with
#  Linf norm ≤ 0     : 88.6 %
 # Linf norm ≤ 0.0003921568627450981: 78.7 %
 # Linf norm ≤ 0.001176470588235294: 64.3 %
 # Linf norm ≤ 0.00392156862745098: 42.0 %
 # Linf norm ≤ 0.01568627450980392: 26.7 %

"FGSM: DeiT - clean_accuracy: 87.5 %"
" Training data-efficient image transformers & distillation through attention"
#robust accuracy for perturbations with
#  Linf norm ≤ 0     : 86.6 %
 # Linf norm ≤ 0.0003921568627450981: 75.6 %
  #Linf norm ≤ 0.001176470588235294: 65.8 %
  #Linf norm ≤ 0.00392156862745098: 55.3 %
  #Linf norm ≤ 0.01568627450980392: 43.5 %
  
"FGSM: Swin"
#clean accuracy:  87.6 %
#robust accuracy for perturbations with
 # Linf norm ≤ 0     : 87.6 %
  #Linf norm ≤ 0.0003921568627450981: 76.6 %
  #Linf norm ≤ 0.001176470588235294: 74.0 %
  #Linf norm ≤ 0.00392156862745098: 67.4 %
  #Linf norm ≤ 0.01568627450980392: 60.9 %
#%%  
"PGD: ResNet"
 # Linf norm ≤ 0     : 62.5 %
 # Linf norm ≤ 0.0003921568627450981: 44.8 %
 # Linf norm ≤ 0.001176470588235294: 17.9 %
 # Linf norm ≤ 0.00392156862745098:  0.4 %
 # Linf norm ≤ 0.011764705882352941:  0.0 %
"PGD: EfficientNet"
#  Linf norm ≤ 0     : 72.2 %
#  Linf norm ≤ 0.0003921568627450981: 61.3 %     ViT > EfficientNet
#  Linf norm ≤ 0.001176470588235294: 43.8 %      ViT > EfficientNet
#  Linf norm ≤ 0.00392156862745098: 12.7 %       EfficientNet > ViT
#  Linf norm ≤ 0.011764705882352941:  0.6 %      EfficientNet > ViT

"PGD: Vit"
#robust accuracy for perturbations with
 #Linf norm ≤ 0     : 81.0 %
  #Linf norm ≤ 0.0003921568627450981: 69.0 %     ViT > EfficientNet
 ## Linf norm ≤ 0.001176470588235294: 44.8 %     ViT > EfficientNet
  #Linf norm ≤ 0.00392156862745098:  7.5 %       EfficientNet > ViT
  #Linf norm ≤ 0.011764705882352941:  0.4 %      EfficientNet > ViT
  
#robust accuracy for perturbations with
 # Linf norm ≤ 0     : 88.6 %
  #Linf norm ≤ 0.0003921568627450981: 78.1 %
  #Linf norm ≤ 0.001176470588235294: 56.7 %
  #Linf norm ≤ 0.00392156862745098: 12.5 %
  #Linf norm ≤ 0.01568627450980392:  0.0 %

"PGD: DeiT"
##  Linf norm ≤ 0     : 79.8 %
  #Linf norm ≤ 0.0003921568627450981: 64.9 %
  #Linf norm ≤ 0.001176470588235294: 37.7 %
  #Linf norm ≤ 0.00392156862745098:  6.7 %
  #Linf norm ≤ 0.011764705882352941:  0.2 %
#%%
"Deepfool"

"FGSM: DeiT > ViT > EfficientNet > ResNet"  
"PGD: EfficientNet > ViT > DeiT  > ResNet"

