import torch
import torch.nn as nn

def accuracy(dataloader, model):  # Dataloader, Model
    correct = 0
    total = 0
    running_loss = 0
    n = len(dataloader)  # batch_size
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    with torch.no_grad():# 1) Grad 
        model.eval()     # 2) Train에서 dropout layer, BatchNorm은Evaluate 할때는 비활성화해줌 
                         #(Evalutaion mode) 평가하는 과정에서는 모든 노드를 사용하겠다는 의미
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            "변경"
            #predicted = outputs.max(1, keepdim=True)[1] # 로그 확률의 최대값을 가지는 인덱스를 얻습니다
            #print(predicted)
            #print(labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item() 
            #print(correct)
            running_loss += loss.item()
            loss_result = running_loss / n   # divide by n (batch_size) 
        #print(outputs)

    acc = 100 * correct / total 
    model.train()  # eval/val 작업이 끝난 후에는 잊지말고 train mode로 모델을 변경 
    return correct,total, acc, loss_result

