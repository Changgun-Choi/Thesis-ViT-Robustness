#!/usr/bin/env python3
"""
A simple example that demonstrates how to run a single attack against
a PyTorch ResNet-18 model for different epsilons and how to then report
the robust accuracy.
"""
import torchvision.models as models
import torch
import numpy as np
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD
import foolbox.attacks as fa
import argparse ##############  Use in Terminal
import timm
assert timm.__version__ == "0.3.2"


def main() -> None:
    parser = argparse.ArgumentParser(description='Vision Transformer')
    parser.add_argument('--model_name', default='resnet18', type=str, help='data name')    
    parser.add_argument('--attack_name', default='LinfPGD', type=str, help='attack name') 
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    #device = "cpu"
    
    
    if args.model_name == 'resnet18':
        # instantiate a model (could also be a TensorFlow or JAX model)
        model = models.resnet18(pretrained=True).eval().to(device) # eval
        
    elif args.model_name == 'efficientnet': 
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True).eval().to(device)

    elif args.model_name == 'vit_B_16_imagenet1k':      
      from pytorch_pretrained_vit import ViT
      model = ViT('B_16_imagenet1k', pretrained=True).eval().to(device)
        
    elif args.model_name == 'deit': ### RuntimeError: Cannot find callable deit_base_distilled_patch16_224 in hubconf
        model = torch.hub.load('facebookresearch/deit:main','deit_base_patch16_224', pretrained=True).eval().to(device)
        
    elif args.model_name =='deit_distilled': ### RuntimeError: Cannot find callable deit_base_distilled_patch16_224 in hubconf
        model = torch.hub.load('facebookresearch/deit:main','deit_base_distilled_patch16_224', pretrained=True).eval().to(device)   
        
    elif args.model_name == 'dino_vit':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').eval().to(device)
 
    elif args.model_name == 'dino_xcit':   
        model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p16').eval().to(device)
        
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=6))
    #labels = torch.tensor(labels,dtype=torch.long)
    #labels = labels.long()    
    #fmodel(images)
    predictions = fmodel(images).argmax(axis=-1)
    predictions
    clean_acc = accuracy(fmodel, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")    
    
    epsilons = [
        0.0,
        0.0002,
        0.0005,
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
        0.1,
       # 0.3,
       # 0.5,
       # 1.0,
    ]
    epsilons = [ 0.0, 0.0002] 
    
    ############# attacks
    if args.attack_name == 'LinfPGD':  # single adversarial attack (Linf PGD)
        attack = LinfPGD()
        raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)
        raw_advs
        clipped_advs
        success
        success.float32()
        success.float32().mean(axis=-1)
        # calculate and report the robust accuracy (the accuracy of the model when it is attacked)
        robust_accuracy = 1 - success.float32().mean(axis=-1) # succes of Attack
        print("robust accuracy for perturbations with")
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
            acc2 = accuracy(fmodel, advs_, labels) 
            print(f"  Linf norm ≤ {eps:<6}: {acc2 * 100:4.1f} %")
            print("    perturbation sizes:")
            perturbation_sizes = (advs_ - images).norms.linf(axis=(1, 2, 3)).numpy()
            print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
            if acc2 == 0:
                break

    elif args.attack_name == 'multiple_attacks': 
        attacks = [
            fa.FGSM(),
            fa.LinfPGD(),
            fa.LinfBasicIterativeAttack(),
            fa.LinfAdditiveUniformNoiseAttack(),
            fa.LinfDeepFoolAttack(),]
        print("epsilons")
        print(epsilons)
        print("")
        
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
#  Linf norm ≤ 0.3   :  0.0 %
#  Linf norm ≤ 0.5   :  0.0 %
#  Linf norm ≤ 1.0   :  0.0 %