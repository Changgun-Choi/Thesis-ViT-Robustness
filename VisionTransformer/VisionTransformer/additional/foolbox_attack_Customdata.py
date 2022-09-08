#!/usr/bin/env python3
#cd "C:/Users/ChangGun Choi/Team Project/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer"
#python foolbox_attack.py --model_name dino_vit --attack_name LinfPGD
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
import argparse ##############  Use in Terminal
import timm
assert timm.__version__ == "0.3.2"
import gc
gc.collect()
torch.cuda.empty_cache()

from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
#%%


def main() -> None:
    parser = argparse.ArgumentParser(description='Vision Transformer')
    parser.add_argument('--model_name', default='resnet18', type=str, help='data name')    
    parser.add_argument('--attack_name', default='LinfPGD', type=str, help='attack name') 
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    device = "cpu"
    
    if args.model_name == 'efficientnet':  # clean accuracy:  87.5 %
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True).eval().to(device)
    
    elif args.model_name == 'vit': # clean accuracy:  81.2 %
        model = timm.create_model('vit_base_patch16_224', pretrained=True).eval()     
        
    elif args.model_name == 'deit': # clean accuracy:  87.5 %
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
        model = deit_base_patch16_224(pretrained=True).eval()
   
    elif args.model_name == 'dino_vit':      # clean accuracy:  0.0 %
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=True).eval().to(device)
        
    elif args.model_name == 'resnet50_dino':       # clean accuracy:  93.8 %
        # instantiate a model (could also be a TensorFlow or JAX model)
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50',pretrained=True).eval().to(device) # eval
 
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)   # normalize inside model. 
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),     # Cifar 10 (32 x 32)  / ImageNet 220 x220 
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    epsilons = [0, 2/255, 4/255, 8/255, .1]
    epsilons = [0, 8/255] 
    #test_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageNet(root='.data/', transform=test_transform), batch_size=16)
    #inputs = inputs.to(device)
    #labels = labels.to(device)
    #images, labels = iter(test_loader).next()
    
    images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=20))  # samples() has only 20 samples and repeats itself if batchsize > 20
    clean_acc = accuracy(fmodel, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")    
    attack = LinfPGD()
    raw_advs, clipped_advs, succ = attack(fmodel, images, labels, epsilons=epsilons)
    
 
    if args.attack_name == 'LinfPGD':  # single adversarial attack (Linf PGD)
        attack = LinfPGD()
        print("epsilons",epsilons)
        success = []
        #clean_acc = accuracy(fmode
        for images,labels in A:  # clipped_advs are not clipped to bounds, but to epsilons
            raw_advs, clipped_advs, succ = attack(fmodel, images.to(device), labels.to(device), epsilons=epsilons)
            success.append(succ.tolist())
            #print(success)
        succ
        torch.LongTensor([success])
        succ
        print(success)
        #succ.float32().mean(axis=-1) # PyTorchTensor(tensor([0.0625, 1.0000],
        robust_accuracy = 1 - succ.float32().mean(axis=-1) # succes of Attackfe (lowering accuracy)
        robust_accuracy
        print("robust accuracy for perturbations with")
        #raw_advs # [0.1972, 0.5189, 0.2180,  ..., 0.5471, 0.0198, 0.3637]
        #clipped_advs # 
        #These are guaranteed to not be perturbed more than epsilon and thus are the actual adversarial examples you want to visualize.
        
        # calculate and report the robust accuracy (the accuracy of the model when it is attacked)
        
        for eps, acc in zip(epsilons, robust_accuracy):
            print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

        "clipped_advs"
        # we will use the clipped advs instead of the raw advs, otherwise 
        # we would need to check if the perturbation sizes are actually within the specified epsilon bound
        print()
        print("we can also manually check this:")
        print()
        print("robust accuracy for perturbations with")
        for eps, advs_ in zip(epsilons, clipped_advs):   # clipped_advs #######################################################
            acc2 = accuracy(fmodel, advs_, labels)       # clip_perturbation(x, xp, epsilon) 
            #fmodel.zero_grad()
            print(f"  Linf norm ≤ {eps:<6}: {acc2 * 100:4.1f} %")
            print("    perturbation sizes:")
            perturbation_sizes = (advs_ - images).norms.linf(axis=(1, 2, 3)).numpy()
            print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
            
            
            if acc2 == 0:
                break
            #%%
#####################################################
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
            

    
if __name__ == "__main__":
    main()
    
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
                

"Single: ResNet"
#clean accuracy:  93.8 %
#robust accuracy for perturbations with
#  Linf norm ≤ 0.0   : 93.8 %
#  Linf norm ≤ 0.0002: 87.5 %
 # Linf norm ≤ 0.0005: 81.2 %
  #Linf norm ≤ 0.0008: 50.0 %
  #Linf norm ≤ 0.001 : 37.5 %
#  Linf norm ≤ 0.0015: 12.5 %
#  Linf norm ≤ 0.002 :  6.2 %
#  Linf norm ≤ 0.003 :  0.0 %
#  Linf norm ≤ 0.01  :  0.0 %
#  Linf norm ≤ 0.1   :  0.0 %

   #output = fmodel(inputs)                       #[16,1000]
 #  output
  # predictions = fmodel(inputs).argmax(axis=-1)  # Index of Max value (value of each 10 labels)
  # predictions  # tensor([841, 814, 408, 772, 364, 377, 675,  59, 285, 675, 473, 675, 167, 375, 675, 814]
  # #accuracy = (predictions == labels).float32().mean()
  # #labels

