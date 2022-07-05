"https://jacobgil.github.io/deeplearning/vision-transformer-explainability"

#1) Attack 전후의 attentino weight k                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 reeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee

#set PYTHONPATH="C:/Users/ChangGun Choi/Team Project/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer"
#cd C:/Users/ChangGun Choi/Team Project/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer
#python vit_explain_FGSM.py --model_name dino_vit --use_cuda --head_fusion "min" --discard_ratio 0.9 

"1. vit_rollout"
#python vit_explain.py --image_path "C:/Users/ChangGun Choi/Team Project/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer/vit_visualization/examples/input.png" --head_fusion "min" --discard_ratio 0.8 

"2. Gradient Attention Rollout for class specific explainability"
#python vit_explain.py --head_fusion "min" --discard_ratio 0.8 --category_index 243
# We can multiply the attention with the gradient of the target class output, and take the average among the attention heads 
# (while masking out negative attentions) to keep only attention that contributes to the target category (or categories).

"Different Attention Head fusion methods"
"Removing the lowest attentions"

import os
os.chdir('C:/Users/ChangGun Choi/Team Project/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer')
import argparse
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from attacks import *
## 
from models import *
from models import VisionTransformer
from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout
####
##
import torchvision.models as models
import torchvision
import torch
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD
import timm
#assert timm.__version__ == "0.3.2"
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
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
    parser.add_argument('--model_name', default='deit', type=str, help='data name') 
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='C:/Users/ChangGun Choi/Team Project/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer/corgie.jpg',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')  # mean/max/min
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    "If category_index isn't specified, Attention Rollout will be used"
   
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    
    args.image_path = "C:/Users/ChangGun Choi/Team Project/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer/corgie.jpg"
    
    #%%
    if args.model_name == 'vit':
        "args_hyper 들을 받아서 정해지는 것들을 maually 지정"
        #model = VisionTransformer(patch_vec_size=48, num_patches=64,latent_vec_dim=128, num_heads=8, mlp_hidden_dim=64, drop_rate=0., num_layers=12, num_classes=10).to(device)
        #model.load_state_dict(torch.load('./model.pth'))            
        model = timm.create_model('vit_base_patch16_224', pretrained=True).eval()     # RuntimeError: The size of tensor a (197) must match the size of tensor b (577) at non-singleton dimension 1
        #elif args.model_name =='vit_finetuned':
        #model.load_from(np.load(args.pretrained_dir))
    
    elif args.model_name == 'deit':    # git clone https://github.com/facebookresearch/deit.git 
        #model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).eval()
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
                
    elif args.model_name == 'dino_vit':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').eval()   
    
    if args.use_cuda:
        model = model.cuda()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  
    ])
    
    img = Image.open(args.image_path)
    input_tensor = transform(img).unsqueeze(0)
    input_tensor
    if args.use_cuda:
        img_tensor = input_tensor.cuda()   # [1,3,224,224]
        

    "추가: FGSM "
    
    #img_tensor = input_tensor.reshape(1,64,-1)  ##############################################################################      
    #img_tensor                        
    "ViT 쓰려면 patches.reshape[batch_size, num_patch, -1] 형태로 만들어야함 "
    img_tensor.requires_grad_(True)
    # 이미지를 모델에 통과시킴
    output = model(img_tensor)

    # 오차값 구하기 (레이블 263은 웰시코기)
    target = torch.tensor([263]).cuda()
    init_pred = output.max(1, keepdim=True)[1] # 로그 확률의 최대값을 가지는 인덱스를 얻습니다
    init_pred
    loss = F.nll_loss(output, target) 
    # 기울기값 구하기
    model.zero_grad()
    loss.backward()
    
    epsilons = [0, 0.1/255, 0.3/255, 1/255, 4/255]  # 기울기의 방향(sign)이 양수인곳에 epsilon 만큼 증가, 음수는 감소
    # 이미지의 기울기값을 추출
    gradient = img_tensor.grad.data            
        # 생성된 적대적 예제를 모델에 통과시킴
        #output = model(perturbed_data)
        #output[:,263]
        # ## 적대적 예제 성능 확인
        
        #perturbed_prediction = output.max(1, keepdim=True)[1]
        #perturbed_prediction_idx = perturbed_prediction.item()
        #perturbed_prediction_name = idx2class[perturbed_prediction_idx]
        #print(output[:,263])
        #print("예측된 레이블 번호:", perturbed_prediction_idx)  # 예측된 레이블 번호: 172
        #print("레이블 이름:", perturbed_prediction_name)   # 레이블 이름: whippet
    #%% Visualize 

    # 시각화를 위해 넘파이 행렬 변환
    for epsilon in epsilons:
        
        original_img_view = img_tensor.squeeze(0).detach().cpu()
        original_img_view = original_img_view.transpose(0,2).transpose(0,1).numpy()
        
        perturbed_data = fgsm_attack(img_tensor, epsilon, gradient) 
        perturbed_data
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
        np_img = img.resize((224,224),Image.ANTIALIAS)
        np_img = np.array(np_img)[:, :, ::-1]
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        mask = show_mask_on_image(np_img, mask)
        cv2.imshow("Input Image", np_img)
        cv2.imshow(name, mask)
        cv2.imwrite("input.png", np_img)
        cv2.imwrite("C:/Users/ChangGun Choi/Team Project/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer/vit_visualization/results/"+ " " + name, mask)
        cv2.waitKey(-1)
