# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 12:19:22 2022

@author: ChangGun Choi
"""
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import vit_explain_FGSM

from PIL import Image
import json

# ## FGSM 공격 함수 정의

def fgsm_attack(image, epsilon, gradient):
    # 기울기값의 원소의 sign 값을 구함
    sign_gradient = gradient.sign()
    # 이미지 각 픽셀의 값을 sign_gradient 방향으로 epsilon 만큼 조절
    perturbed_image = image + epsilon * sign_gradient
    # [0,1] 범위를 벗어나는 값을 조절
    "변경" #########################################################################################################################
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 작은 변화가 적용된 이미지를 리턴합니다
    return perturbed_image 


def attack(model, device, attackloader, epsilon ):

    # 정확도 카운터
    correct = 0
    adv_examples = []
    final = []

    # 테스트 셋의 모든 예제에 대해 루프를 돕니다 
    "Batch_size 1씩"
      
    for data, target in attackloader:
        # 디바이스(CPU or GPU) 에 데이터와 라벨 값을 보냅니다
        data, target = data.to(device), target.to(device)
        # 텐서의 속성 중 requires_grad 를 설정합니다. 공격에서 중요한 부분입니다
        data.requires_grad = True
        
        "변경: Accuracy가 다른 이유는 eval()"
        model.eval() 
        
        output, attention_list = model(data)       # Compare Attention MAP
        "변경"
        #_, init_pred = torch.max(output, 1)
        init_pred = output.max(1, keepdim=True)[1] # 로그 확률의 최대값을 가지는 인덱스를 얻습니다

        # 만약 초기 예측이 틀리면, 공격하지 않도록 하고 계속 진행합니다
        if init_pred.item() != target.item():  
            #"추가해줌"
                    ########################################3
            continue                            # 처음으로 돌아감 

        "여기부터 공격 시작"
        # 손실을 계산합니다
        loss = F.nll_loss(output, target)

        # 모델의 변화도들을 전부 0으로 설정합니다
        model.zero_grad()

        # 후방 전달을 통해 모델의 변화도를 계산합니다
        loss.backward()

        # 변화도 값을 모읍니다
        data_grad = data.grad.data

        # FGSM 공격을 호출합니다
        perturbed_data = fgsm_attack(data, epsilon, data_grad)     # 공격받은 Perturbed

        # 작은 변화가 적용된 이미지에 대해 재분류합니다
        output, attention_attack = model(perturbed_data)           # Compare Attention MAP
        #print(output)
        "Compare Attention MAP"
        
        # 올바른지 확인합니다
        final_pred = output.max(1, keepdim=True)[1] # 로그 확률의 최대값을 가지는 인덱스를 얻습니다
        if final_pred.item() == target.item():
            correct += 1
            # 0 엡실론 예제에 대해서 저장합니다
            if (epsilon == 0) and (len(adv_examples) < 5):                ## 5개씩 저장 
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()  # perturbed_data
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex)) 
        else:
            # 추후 시각화를 위하 나머지 전부 예제들을 저장합니다
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        
        final.append(final_pred.item()) 
        
    #print(output)    
    #print(final)
    # 해당 엡실론에서의 최종 정확도를 계산합니다
    final_acc = correct/float(len(attackloader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(attackloader), final_acc))

    # 정확도와 적대적 예제를 리턴합니다
    return final_acc, adv_examples

