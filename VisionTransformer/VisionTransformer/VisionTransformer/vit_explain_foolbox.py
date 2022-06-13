"https://jacobgil.github.io/deeplearning/vision-transformer-explainability"
#set PYTHONPATH="C:/Users/ChangGun Choi/Team Project/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer"
#cd C:/Users/ChangGun Choi/Team Project/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer
#python vit_explain_foolbox.py --model_name efficient --attack_name LinfPGD --visual Grad_Cam --use_cuda True

"1. vit_rollout"
#python vit_explain.py --image_path "C:/Users/ChangGun Choi/Team Project/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer/vit_visualization/examples/input.png" --head_fusion "min" --discard_ratio 0.8 

"2. Gradient Attention Rollout for class specific explainability"
#python vit_explain.py --head_fusion "min" --discard_ratio 0.8 --category_index 243
# We can multiply the attention with the gradient of the target class output, and take the average among the attention heads 
# (while masking out negative attentions) to keep only attention that contributes to the target category (or categories).

"Different Attention Head fusion methods"
"Removing the lowest attentions"
#!pip install grad-cam
import os
#os.chdir('C:/Users/ChangGun Choi/Team Project/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer')
import argparse
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
#from attacks import *                                                                                                                                           
## 
from models import *
from models import VisionTransformer
from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout
####
##
import torchvision.models as models
import torch
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD
#from pytorch_pretrained_vit import ViT
import foolbox as fb
import timm
#assert timm.__version__ == "0.3.2"

from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
#%%
"Grad Cam"
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from grad_cam import GradCam,GuidedBackpropReLUModel,show_cams,show_gbs,preprocess_image
#https://www.kaggle.com/code/meaninglesslives/efficientnet-gradcam-comparison-to-other-models/notebook
#%%
def get_args(): 
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")
    return args

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--attack_name', default='LinfPGD', type=str, help='attack name') 
    parser.add_argument('--model_name', default='vit', type=str, help='data name') 
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='C:/Users/ChangGun Choi/Team Project/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer/Grad-CAM-pytorch/input.png',
                        help='Input image path') 
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')  # mean/max/min
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    
    parser.add_argument('--visual', type=str, default='Grad_Cam')

    "If category_index isn't specified, Attention Rollout will be used"
   
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    #args.image_path = 'C:/Users/ChangGun Choi/Team Project/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer/Grad-CAM-pytorch/input.png'
    
    #%%
    with torch.no_grad():
        "https://rwightman.github.io/pytorch-image-models/models/vision-transformer/"
        if args.model_name == 'resnet_18':   # 68.4 %
            model = torchvision.models.resnet18(pretrained=True).eval().to(device)
            
        elif args.model_name == 'resnet50': # 80.53
            model = timm.create_model('resnet50',  pretrained=True).to(device)
          # ResNet50-swsl pre-trained on IG-1B-Targeted (Mahajan et al. (2018)) using semi-weakly supervised methods (Yalniz et al. (2019))
        #elif args.model_name == 'mobilenet3':
         #   model = timm.create_model('mobilenetv3_large_100',  pretrained=True).eval().to(device)
        elif args.model_name == 'VGG':  # 80
            model = models.vgg19(pretrained=True).eval().to(device)  # Not Eval
            blob_name = 'features'
            target_layer_names=['4','20','36']
            activation_layer_name = 'ReLU'
            
        elif args.model_name == 'efficient':  # 80
            #model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True).eval().to(device)
            #model = timm.create_model('efficientnet_b0', pretrained=True)
            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_pretrained('efficientnet-b0').eval().to(device)
            blob_name = '_blocks'
            target_layer_names=['1','10','15']
            activation_layer_name = 'MemoryEfficientSwish'
      
        elif args.model_name == 'vit':
            model = timm.create_model('vit_base_patch16_224', pretrained=True).eval().to(device)  
            
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
            
    if args.use_cuda:
        model = model.cuda()
#%%
    preprocessing = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-3)
    #preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    "MODEL - .cuda() "
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)  # Defining shape of data to be injected into Model # bounds of the input space (before the preprocessing).
    #images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=1)) # [1,3,224,224] Batchsize = 1 for Explain
    #images, labels = fb.utils.samples(fmodel, dataset='imagenet', batchsize=1)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])
    
    img = Image.open(args.image_path)
    #img = img.resize((224, 224))     # [224,224]
    images = transform(img).unsqueeze(0)  # [1,3,224,224]  # 3: RGB    
    #"label is added"
    labels = torch.tensor([263])             #
    
    if args.use_cuda:
        images = images.cuda() # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
        labels = labels.cuda()
   
    epsilons = [0, 0.1/255, 0.3/255, 1/255, 4/255]
    #%% "Attack" 
    #######################################################################################################################
    if args.attack_name == 'LinfPGD':  # single adversarial attack (Linf PGD)
        attack = LinfPGD()
        print("epsilons")
        print(epsilons)
        #images = inputs
        raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)
        
        "calculate and report the robust accuracy (the accuracy of the model when it is attacked)"
        for epsilon, advs_ in zip(epsilons, clipped_advs): # "clipped_advs" we would need to check if the perturbation sizes are actually within the specified epsilon bound
            original_img_view = images.squeeze(0).detach().cpu()  #[3,224,224]
            original_img_view = original_img_view.transpose(0,2).transpose(0,1).numpy()  #[224,224,3]
            "clipped_advs"
            perturbed_data = advs_  # advs_
            perturbed_data_view = perturbed_data.squeeze(0).detach().cpu()
            perturbed_data_view = perturbed_data_view.transpose(0,2).transpose(0,1).numpy()
        
            #plt.imshow(perturbed_data_view)
            # ## 원본과 적대적 예제 비교
            f, a = plt.subplots(1, 2, figsize=(10, 10))
            # 원본
            #a[0].set_title(prediction_name)
            #a[0].imshow(original_img_view)        
            #a[1].set_title(perturbed_prediction_name)
            #a[1].imshow(original_img_view)
            #plt.show()
            
            if args.visual == 'Grad_Cam':
                #advs_ = iter(clipped_advs).__next__()  # For one Testset
                "Grad_Cam - model.cpu(): use_cuda=False"
                model = model.cpu()  
                advs_ex = advs_.cpu() 
              
                grad_cam = GradCam(model=model, blob_name = blob_name, target_layer_names=target_layer_names, use_cuda=False)
                img = cv2.imread(args.image_path, 1)             
                img = np.float32(cv2.resize(img, (224, 224))) / 255
                inputs = preprocess_image(img) 
                
                #advs_ex.shape # # torch.Size([1, 3, 224, 224])
                #inputs.shape # torch.Size([1, 3, 224, 224])
                #advs_ex - inputs
                "Adversarial"
                inputs = advs_ex
                # If None, returns the map for the highest scoring category. Otherwise, targets the requested index.
                target_index = None
                mask_dic = grad_cam(inputs, target_index)
                # Model, inputs, use_cuda=False -> CPU: Otherwise, Error Expected 4-dimensional input for 4-dimensional weight
                show_cams(img, mask_dic, epsilon, args.model_name)
                
                #gb_model = GuidedBackpropReLUModel(model=model, activation_layer_name = activation_layer_name, use_cuda=False)
                #show_gbs(inputs, gb_model, target_index, mask_dic)
            
            elif args.visual == 'Grad_Cam_2':
               #############################################
                #target_layer = [model._conv_head] # efficient
                #target_layers = [model.features.conv]
                target_layer = [model.layer4[-1]]  # resnet50
                input_tensor = inputs.cpu()
                model = model.cpu() 

                #rgb_img = np.float32(img) / 255
                import cv2
                rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]          
                rgb_img = np.float32(rgb_img) / 255
                #rgb_img = np.float32(cv2.resize(rgb_img, (224, 224))) / 255
                input_tensor = preprocess_image(rgb_img).cpu()
                #input_tensor  = images[model.layer4[-1]]
                targets = [ClassifierOutputTarget(263)]
                
                # Construct the CAM object once, and then re-use it on many images:
                cam = GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda)
                #rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
                #rgb_img = np.float32(rgb_img) / 255
                #rgb_img.resize((224, 224,3))
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
 
                # In this example grayscale_cam has only one image in the batch:
                grayscale_cam = grayscale_cam[0, :]     
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                Image.fromarray(visualization)    
                    
            elif args.visual == 'ViT_vis':
                            
                "Original Roll_out"
               # perturbed_data    
                if args.category_index is None:          # "If category_index isn't specified, Attention Rollout will be used"
                    print("Doing Attention Rollout")
                    #attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, discard_ratio=args.discard_ratio)
                    #mask = attention_rollout(perturbed_data)
                    
                    attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, discard_ratio=args.discard_ratio)
                    mask = attention_rollout(perturbed_data) ###############
                    
                    name = "attention_rollout_{}_{:.3f}_{}_{}.png".format(args.model_name, args.discard_ratio, args.head_fusion, epsilon)
                else:
                    print("Doing Gradient Attention Rollout")
                    grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
                    mask = grad_rollout(perturbed_data, args.category_index)
                    name = "grad_rollout_{}_{}_{:.3f}_{}_{}.png".format(args.model_name, args.category_index, args.discard_ratio, args.head_fusion, epsilon)
                # Roll_out for Adversal Examples
                np_img = img.resize((224,224),Image.ANTIALIAS)
                np_img = np.array(img)[:, :, ::-1]
                mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
                mask = show_mask_on_image(np_img, mask)
                #cv2.imshow("Input Image", np_img)
                cv2.imshow(name, mask)
                #cv2.imwrite("input.png", np_img)
                cv2.imwrite("C:/Users/ChangGun Choi/Team Project/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer/results"+ " " + name, mask)
                #"./vit_visualization"+ " " + name
                cv2.waitKey(-1)   
            
    
                 
#%%
    elif args.attack_name == 'multiple_attacks': 
        #images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=1))  ############### Batchsize = 1 for Explain
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
            
        "VIT_Explain part"
        # 시각화를 위해 넘파이 행렬 변환            

        original_img_view = img_tensor.squeeze(0).detach().cpu()
        original_img_view = original_img_view.transpose(0,2).transpose(0,1).numpy()
        
        perturbed_data = fgsm_attack(img_tensor, epsilon, gradient) 
        perturbed_data_view = perturbed_data.squeeze(0).detach().cpu()
        perturbed_data_view = perturbed_data_view.transpose(0,2).transpose(0,1).numpy()
    
        #plt.imshow(perturbed_data_view)
    
        # ## 원본과 적대적 예제 비교
        f, a = plt.subplots(1, 2, figsize=(10, 10))
        # 원본
        #a[0].set_title(prediction_name)
        a[0].imshow(original_img_view)
    
        # 적대적 예제
        #a[1].set_title(perturbed_prediction_name)
        a[1].imshow(perturbed_data_view)
        plt.show()
        
        "Original Roll_out"
       # perturbed_data    
        if args.category_index is None:          # "If category_index isn't specified, Attention Rollout will be used"
            print("Doing Attention Rollout")
            #attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, discard_ratio=args.discard_ratio)
            #mask = attention_rollout(perturbed_data)
            
            attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, discard_ratio=args.discard_ratio)
            mask = attention_rollout(perturbed_data) ###############
            
            name = "attention_rollout_{}_{:.3f}_{}_{}.png".format(args.model_name, args.discard_ratio, args.head_fusion, epsilon)
        else:
            print("Doing Gradient Attention Rollout")
            grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
            mask = grad_rollout(perturbed_data, args.category_index)
            name = "grad_rollout_{}_{}_{:.3f}_{}_{}.png".format(args.model_name, args.category_index, args.discard_ratio, args.head_fusion, epsilon)

        # Roll_out for Adversal Examples
        
        np_img = np.array(img)[:, :, ::-1]
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        mask = show_mask_on_image(np_img, mask)
        #cv2.imshow("Input Image", np_img)
        cv2.imshow(name, mask)
        #cv2.imwrite("input.png", np_img)
        cv2.imwrite("C:/Users/ChangGun Choi/Team Project/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer/vit_visualization/"+ " " + name, mask)
        "./vit_visualization"+ " " + name
        cv2.waitKey(-1)
        
        
#%%

             
#RuntimeError: The size of tensor a (197) must match the size of tensor b (577) at non-singleton dimension 1