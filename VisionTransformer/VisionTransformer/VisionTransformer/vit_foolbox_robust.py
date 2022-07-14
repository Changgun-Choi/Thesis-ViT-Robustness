"Server"
# git stash
# git pull
# conda activate thesis
# cd /home/cchoi/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer
# python vit_foolbox_robust.py --model_name vit_s --attack_name PGD --batch_size 16 --data_divide 10 --data_path server 

#!/usr/bin/env python3
#cd "C:/Users/ChangGun Choi/Team Project/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer"
#python vit_foolbox_robust.py --model_name efficient --attack_name FGSM --batch_size 16 --data_divide 1000 
#python vit_foolbox_robust.py --model_name efficient --attack_name PGD --batch_size 16 --data_divide 10 --data_path server
#python vit_foolbox_robust.py --model_name VGG --attack_name PGD --batch_size 16 --data_divide 10 --data_path server 

# nvidia-smi
""" A simple example that demonstrates how to run a single attack against a PyTorch ResNet-18 model for different epsilons and how to then report
the robust accuracy """
#!pip install grad-cam
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
from tqdm.notebook import tqdm
#%%
#import json
#CLASSES = json.load(open('C:/Users/ChangGun Choi/Team Project/Thesis_Vision/Adversarial_attack/imagenet_classes.json'))
#len(CLASSES)
#idx2class = [CLASSES[str(i)] for i in range(1000)]
#idx2class

def softmax_activation(inputs): 
    inputs = inputs.tolist()
    exp_values = np.exp(inputs - np.max(inputs)) 
    
    # Normalize 
    probabilities = exp_values / np.sum(exp_values)
    return probabilities 

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vision Transformer')
    parser.add_argument('--model_name', default='vit', type=str, help='data name')    
    parser.add_argument('--attack_name', default='LinfPGD', type=str, help='attack name') 
    parser.add_argument('--batch_size',default = 16, type = int)
    parser.add_argument('--data_divide',default = 10, type = int, help = 'multiply by 0.01')  # /10 : 5000  args.data_path 
    parser.add_argument('--data_path',default = 'local', type = str) 
    parser.add_argument('--stepsize',default = '4', type = int) # /4, 8, 12
    parser.add_argument('--PGD_change',default = 'no', type = str) 
    #parser.add_argument('--epsilon',default =  0.001176, type = float)  # 0.3/255
    "Define args"
    args = parser.parse_args()  
    #print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   ############################################
    #device = "cpu"
    with torch.no_grad():
        "https://rwightman.github.io/pytorch-image-models/models/vision-transformer/"
        if args.model_name == 'resnet_18':   # 68.4 %
            model = torchvision.models.resnet18(pretrained=True).eval().to(device)
            
        elif args.model_name == 'resnet50': # 80.53
            model = timm.create_model('resnet50',  pretrained=True).eval().to(device)
          # ResNet50-swsl pre-trained on IG-1B-Targeted (Mahajan et al. (2018)) using semi-weakly supervised methods (Yalniz et al. (2019))
        #elif args.model_name == 'mobilenet3':
         #   model = timm.create_model('mobilenetv3_large_100',  pretrained=True).eval().to(device)
        elif args.model_name == 'VGG':  # 80
            model = models.vgg19(pretrained=True).eval().to(device) 
        
        elif args.model_name == 'efficient':  #  80.4
            model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True).eval().to(device)
            #model = timm.create_model('efficientnet_b0', pretrained=True)
            #from efficientnet_pytorch import EfficientNet
            #model = EfficientNet.from_pretrained('efficientnet-b0').eval().to(device)
        elif args.model_name == 'efficient_b4':
            model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True).eval().to(device)
   
        elif args.model_name == 'vit':
            model = timm.create_model('vit_base_patch16_224', pretrained=True).eval().to(device)  
        elif args.model_name == 'vit_s':
             model = timm.create_model('vit_small_patch32_224', pretrained=True).eval().to(device)      
            
             #  ImageNet-1k weights fine-tuned from in21k @ 224x224
        elif args.model_name == 'vit_L':
            model = timm.create_model('vit_large_patch16_224', pretrained=True).eval().to(device)  
             
            
        elif args.model_name == 'vit_hybrid': #   
            "Hybrid Vision Transformers "  # https://github.com/xinqi-fan/ABAW2021/blob/main/models/vision_transformer_hybrid.py
            model = timm.create_model('vit_large_r50_s32_224', pretrained=True,num_classes=1000).eval().to(device)
            # vit_base_r50_s16_384  # vit_base_resnet50_384
            
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
       
        elif args.model_name == 'swin':
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
            timm.list_models('*resnet15*')
       
            #%%
 
    preprocessing = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-3)   # normalize inside model.  
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),     # Cifar 10 (32 x 32)  / ImageNet 220 x220 
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])    
    
    epsilons = [0, 0.1/255, 0.3/255, 1/255, 4/255]  # Maximum perturbation
    
    #epsilons = [0, 0.1/255, 0.3/255, 0.5/255, 0.8/255, 1/255, 4/255]  # 0.5/255, 0.8/255
    
    if args.data_path == 'local':
        data_path = 'C:/Users/ChangGun Choi/Team Project/Thesis_data/val'
    elif args.data_path == 'server':
        data_path = '/home/cchoi/Thesis_data/data/val'
        
    testset = torchvision.datasets.ImageNet(data_path, split='val', transform=test_transform)
    sample = list(range(0, len(testset), args.data_divide))   # 16 * 3125 * 0.1 : 5000
    print(len(testset))
    print(len(sample)) # 5000
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
       #%%
    if args.attack_name == 'PGD':  # FGSM을 step 단위로 나눠서 사용하는 방식의 공격
        "default settinig for steps"
        if args.PGD_change == 'no': 
            attack = LinfPGD()  # LinfPGD = LinfProjectedGradientDescentAttack # Distance Measure : Linf
            accuracy = 0 
            success = torch.zeros(len(epsilons),args.batch_size).to(device)
            print(len(val_loader))
            for batch_idx, (image, label) in enumerate(val_loader):
                print("Attack: {}/{}".format(batch_idx+1, len(val_loader)-1))
                images = image.to(device)
                labels = label.to(device)
           
                #images, labels = ep.astensors(images, labels)
                clean_acc = get_acc(fmodel, images, labels)
                raw_advs, clipped_advs, succ = attack(fmodel, images, labels, epsilons=epsilons) 
                for epsilon, advs_ in zip(epsilons, clipped_advs): # "clipped_advs" we would need to check if the perturbation sizes are actually within the specified epsilon bound
                    #advs_ = advs_.requires_grad_(True)    
                    output = fmodel(advs_)
             
                    
                    #perturbed_prediction = output.max(1, keepdim=True)[1]
                    #perturbed_prediction_idx = perturbed_prediction.item()
                    #perturbed_prediction_name = idx2class[perturbed_prediction_idx]
                    #print(output[:,263])
                    #accuracy = np.max(softmax_activation(output), axis=1)
                    #accuracy = round(accuracy[0], 2)
                    #print("Confidence: {}%_ {}".format(accuracy * 100, perturbed_prediction_name)) 
                    
                succ = torch.cuda.FloatTensor(succ.detach().cpu().numpy()) # 1) EagerPy -> numpy 2) Numpy -> FloatTensor)
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
        else: 
            "Different step_size"
            "defaults to 0.01 / 0.3"
            #stepsize = [i/4 for i in eps] 
            #epsilons = [args.epsilon]  # list
            epsilons = [0, 0.1/255, 0.3/255, 0.5/255, 0.8/255, 1/255]
            #epsilons = []
            accuracy = 0 
            r_success = torch.zeros(len(epsilons),args.batch_size).cuda()
            for batch_idx, (image, label) in enumerate(val_loader):
                print("Attack: {}/{}".format(batch_idx+1, len(val_loader)-1))
                success = torch.zeros(len(epsilons),args.batch_size).cuda()
                for i, eps in enumerate(epsilons):
                    stepsize = eps/args.stepsize # learning rate
                    attack = LinfPGD(rel_stepsize=0.033, abs_stepsize=stepsize, steps=40, random_start=True) # Whether the perturbation is initialized randomly 
                           # abs_stepsize: If given, it takes precedence over rel_stepsize, # rel_stepsize: Stepsize relative to epsilon
                    images = image.cuda()
                    labels = label.cuda() 
                    #images, labels = ep.astensors(images, labels)
                    #clean_acc = get_acc(fmodel, images, labels)
                    "attack"
                    raw_advs, clipped_advs, succ = attack(fmodel, images, labels, epsilons=eps)    
                    #print(succ)
                    success[i] = torch.cuda.FloatTensor(succ.detach().cpu().numpy()) # 1) EagerPy -> numpy 2) Numpy -> FloatTensor)
                    #print(success)
                r_success += success
                #print(r_success)
            accuracy = accuracy / len(val_loader)
            print(f"clean accuracy:  {accuracy * 100:.1f} %") 
            r_success = r_success/len(val_loader)            #  # succes of Attack (lowering accuracy)
            robust_accuracy = 1 - r_success.mean(dim = -1)     # t.mean(dim=1): Mean of last dimension (different with other dim)
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
        for eps, acc in zip(epsilons, robust_accuracy):
            print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")   
        plt.figure(figsize=(5,5))
        plt.plot(epsilons, robust_accuracy.cpu().numpy()) 
        plt.show()
         
        
    elif args.attack_name == 'Deepfool': 
        "steps=50"
        
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
            

#%%
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







