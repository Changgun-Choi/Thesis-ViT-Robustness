"Main py"
#cd "C:/Users/ChangGun Choi/Team Project/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer"
#python vit.py --pretrained 1 --mode test
import os
os.chdir('C:/Users/ChangGun Choi/Team Project/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer')

# Module Created (py)
import patchdata
import model
import test
import attacks  ## 
########################
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

import argparse ##############  Use in Terminal
import time
import datetime      # Calculate execution Time
from vit_explain import *
start = time.time()
torch.manual_seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vision Transformer')                    
    parser.add_argument('--img_size', default=32, type=int, help='image size')
    parser.add_argument('--patch_size', default=4, type=int, help='patch size')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')   
    parser.add_argument('--save_acc', default=50, type=int, help='val acc')               # Start Saving from 50% accuracy
    parser.add_argument('--epochs', default=501, type=int, help='training epoch')
    parser.add_argument('--lr', default=2e-3, type=float, help='learning rate')
    parser.add_argument('--drop_rate', default=.1, type=float, help='drop rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes')  # Important (Number of Nodes of Classification)
    parser.add_argument('--latent_vec_dim', default=128, type=int, help='latent dimension')
    parser.add_argument('--num_heads', default=8, type=int, help='number of heads')
    parser.add_argument('--num_layers', default=12, type=int, help='number of layers in transformer')
    parser.add_argument('--dataname', default='cifar10', type=str, help='data name')      # ImageNet
    parser.add_argument('--model_name', default='vit', type=str, help='data name')       # Vit
    "If function"
    parser.add_argument('--mode', default='train', type=str, help='train or evaluation')  # if args.mode == 'train': 이렇게 조건을 지정
    parser.add_argument('--pretrained', default=0, type=int, help='pretrained model')     # if args.pretrained == 1
    parser.add_argument('--optim', default = 'Adam', type =str, help='Optimizer')
    args = parser.parse_args()
    print(args)   # check hyper-parameter setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    "Load Data: Image Patches-args 4개 여기서 사용"
    d = patchdata.Flattened2Dpatches(dataname=args.dataname, img_size=args.img_size, patch_size=args.patch_size, batch_size=args.batch_size)                                    
    trainloader, valloader, testloader, attackloader = d.patchdata()   # attack_loader
    image_patches, _ = iter(trainloader).next()  # image_patches.size를 사용하려고 하나 꺼내옴  
                        # for문처럼 하나씩 가져와서 처리하고 폐기해서 대용량처리에 좋음 # 함수의 경우: generator(), yield() : 메모리에 저장을 하지 않음, yield는 함수의 return 과 비슷"
    "Load Model: ViT"
    # hyper for ViT model
    latent_vec_dim = args.latent_vec_dim   # mlp hidden을 정의하는데 사용하기 위해서 args.불러옴
    mlp_hidden_dim = int(latent_vec_dim/2) # 임의로 설정한것: 절반
    num_patches = int((args.img_size *args.img_size ) /( args.patch_size * args.patch_size))  # (img_size * img) 에서 각각을 patch_size 로 나누면 전체 갯수   

    if args.model == 'vit':
        model = model.VisionTransformer(patch_vec_size=image_patches.size(2), num_patches=image_patches.size(1),   # image_patches.size 
                                  latent_vec_dim=latent_vec_dim, num_heads=args.num_heads, mlp_hidden_dim=mlp_hidden_dim,
                                  drop_rate=args.drop_rate, num_layers=args.num_layers, num_classes=args.num_classes).to(device)
        
    
    # vit(image_patches.to(device))
     # Test 40 * 128
#%% 
    "Train"
    if args.pretrained == 1:
      if args.model == 'vit':
          model.load_state_dict(torch.load('./model.pth'))   # 바뀔수 있음
        
      if args.model == 'vit_B_16_imagenet1k':  
        
        from pytorch_pretrained_vit import ViT
        model = ViT('B_16_imagenet1k', pretrained=True)
        model = model.eval().to(device)
                  
          
      elif args.model_name == 'deit':
          model = torch.hub.load('facebookresearch/deit:main','deit_tiny_patch16_224', pretrained=True)
          model = model.eval().to(device)
          
      elif args.model_name =='deit_distilled':
          model = torch.hub.load('facebookresearch/deit:main','deit_base_distilled_patch16_224', pretrained=True)
          model = model.eval().to(device)
          
      elif args.model_name == 'efficientnet':
      
          model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
          #utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
          model.eval().to(device)
          model = model.eval().to(device)
      elif args.model_name == 'dino_vit':
          model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
          model = model.eval().to(device)
          
      elif args.model_name == 'dino_xcit':   
          model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p16')
          model = model.eval().to(device)
        
    if args.mode == 'train':   # hyper-parameter
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss() # classification
        
        if  args.optim == 'Adam':  # hyper
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optim == 'SGD':  # else 는 'SGD' 이렇게 설정이 불가능함 
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)  # Finetuning 할때 사용함 (논) 
        #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(trainloader), epochs=args.epochs)

        # Train
        n = len(trainloader)                 # Num of batch
        best_acc = args.save_acc             # 0.5: Start Saving from 50% accuracy
        for epoch in range(args.epochs):     # args.epochs
        
            running_loss = 0
            for img, labels in trainloader:  # Flattened patch
                optimizer.zero_grad()        # 먼저 grad zero로 만들어
                class_outputs, __ = model(img.to(device))  #1) Liner projection, position embedding -> Trnasformer -> 12 Head -> output
                # img = image_patches: [128, 64, 48]
                
                loss = criterion(class_outputs, labels.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()  # loss.items(): 텐서에서 단순 scalar 값을 return 해줌
                #scheduler.step()

            train_loss = running_loss / n    # 평균 loss 임 (n: len(trainloader)) 
            val_acc, val_loss = test.accuracy(valloader, model)  # Dataloader, Model
            # if epoch % 5 == 0:
            print('[%d] train loss: %.3f, validation loss: %.3f, validation acc %.2f %%' % (epoch, train_loss, val_loss, val_acc)) 
            # train, val의 차이를 결과보면서 normalization을 강하게 할지 고민  

            if val_acc > best_acc:  # 작으면 저장이 안되니까 Earlystopping 
                best_acc = val_acc # best 보다 크면 replace
                print('[%d] train loss: %.3f, validation acc %.2f - Save the best model' % (epoch, train_loss, val_acc))
                torch.save(model.state_dict(), './model.pth')  # 계속 update 되는 것 

#%% Test(Accuracy, Adversairal Attack)"
    else:
        #vit.load_state_dict(torch.load('./model.pth'))   
        
        test_acc, test_loss = test.accuracy(testloader, model)  # vit: Trained model
        print('test loss: %.3f, test acc %.2f %%' % (test_loss, test_acc))
        
        ## Test Attactks  
        "attackloader:공격을 하나씩 해야해서 batch_size = 1"
        # Sampled data
        
        accuracies = []
        examples = []
        #epsilons = [0, .05, .1, .15, .2, .25, .3]
        epsilons = [.05, .1,.3]
        # 각 엡실론에 대해 테스트 함수를 실행합니다
        for eps in epsilons:
            acc, ex = attacks.attack(vit, device, attackloader, eps) # "attackloader" 
            accuracies.append(acc)
            examples.append(ex)
        
        # examples(ex) : init_pred.item(), final_pred.item(), adv_ex 
        
        plt.figure(figsize=(5,5))
        plt.plot(epsilons, accuracies, "*-")
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.xticks(np.arange(0, .35, step=0.05))
        plt.title("Accuracy vs Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.show()
        #%%
        
        cnt = 0
        plt.figure(figsize=(8,10))
        for i in range(len(epsilons)):
            for j in range(len(examples[i])):
                cnt += 1
                plt.subplot(len(epsilons),len(examples[0]),cnt)
                plt.xticks([], [])
                plt.yticks([], [])
                if j == 0:
                    plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
                orig,adv,ex = examples[i][j]  # examples[epsilon][5 examples]
                plt.title("{} -> {}".format(orig, adv))
                plt.imshow(ex)
                #plt.imshow(ex, cmap="gray")

                ###################################  ex의 형태가 중요
            

                
        plt.tight_layout()
        plt.show()
        
        # test loss: 0.501, test acc 84.64 %
        
        now = time.time()
        sec = now-start
        
        times = str(datetime.timedelta(seconds=sec)).split(".")
        times = times[0]
        print(times)     # Count time 

#%%

#test loss: 0.501, test acc 84.64 %
#Epsilon: 0      Test Accuracy = 165 / 500 = 0.33
#Epsilon: 0.05   Test Accuracy = 136 / 500 = 0.272
#Epsilon: 0.1    Test Accuracy = 92 / 500 = 0.184
#Epsilon: 0.15   Test Accuracy = 83 / 500 = 0.166
#Epsilon: 0.2    Test Accuracy = 81 / 500 = 0.162
#Epsilon: 0.25   Test Accuracy = 59 / 500 = 0.118
#Epsilon: 0.3    Test Accuracy = 61 / 500 = 0.122
"vit_explain.py"
                #transform = transforms.Compose([
                #    transforms.Resize((224, 224)),
                #    transforms.ToTensor(),
                #    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                #])
                
               # img = ex.resize((224, 224))
               # input_tensor = transform(ex).unsqueeze(0)
                #args.category_index is None:          # "If category_index isn't specified, Attention Rollout will be used"
#                    print("Doing Attention Rollout")
               # attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, discard_ratio=args.discard_ratio)
               # mask = attention_rollout(input_tensor)
               # name = "attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion) 