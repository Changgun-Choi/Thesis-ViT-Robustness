import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

#%%
class PatchGenerator:   # Self-Created: Patch 만드는 거를 Pre-processing으로 아에 처리한것 

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, img):
        num_channels = img.size(0)  # Channel, height, width(batch not added yet)
                     # Unfold: 세로로 한번,                       #가로로 한번 자름: # x.unfold(dimension, size of each slice, stride)
        patches = img.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size).reshape(num_channels, -1, self.patch_size, self.patch_size)  
                     # 3x (16x16) x (16x16) : chanel x (가로,세로 갯수) x (patch size x patch size)                       # -1: flatten many patches into 1 row (256)
        patches = patches.permute(1,0,2,3)  # Reform: index 0, 1(number of patches: 256)
        num_patch = patches.size(0)
        return patches.reshape(num_patch,-1) # Flatten each patch (두번째 flatten)
 
class Flattened2Dpatches:

    def __init__(self, patch_size=8, dataname='imagenet', img_size=32, batch_size=64): # Hyper-parameter : args
        self.patch_size = patch_size  
        self.dataname = dataname
        self.img_size = img_size
        self.batch_size = batch_size

    def make_weights(self, labels, nclasses):     # labels, 와 nclasses는 data를 불러오고나서 정해지는 거니까 hyper가 아님 (self. 이용할 필요 x)
        # 각 클래스가 동일하게 Sample 되도록 Weight 를 만들어줌
        labels = np.array(labels)
        weight_arr = np.zeros_like(labels)
        _, counts = np.unique(labels, return_counts=True)
        for cls in range(nclasses):
            weight_arr = np.where(labels == cls, 1/counts[cls], weight_arr) 
    
        return weight_arr 

    def patchdata(self):    
        mean = (0.4914, 0.4822, 0.4465)  # Test 할때는 사용 x 
        std = (0.2023, 0.1994, 0.2010)   # Test 할때는 사용 x 
        # Train
        train_transform = transforms.Compose([transforms.Resize(self.img_size), transforms.RandomCrop(self.img_size, padding=2),
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std),  # Mean, std for training
                                              PatchGenerator(self.patch_size)]) ################################## PatchGenerator
        
        # Test: No need to crop, augmenation as training
        test_transform = transforms.Compose([transforms.Resize(self.img_size), transforms.ToTensor(),
                                             transforms.Normalize(mean, std)]) ## PatchGenerator for Transformer
      
        #attack_transform = transforms.Compose([transforms.Resize(self.img_size), transforms.ToTensor(),
                                           #  PatchGenerator(self.patch_size)])
        
        if self.dataname == 'cifar10':  # Hyper-parameter
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
            attackset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
            #attackset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
            evens = list(range(0, len(testset), 2))  # [0, 2, 4, 6, 8]
            odds = list(range(1, len(testset), 2))   # [1, 3, 5, 7, 9]: interval(2)
            valset = torch.utils.data.Subset(testset, evens)   # Subset: Even number
            testset = torch.utils.data.Subset(testset, odds)   # Odds number
            ##################
            #attack = list(range(0, len(testset), 10)) # Test 40 * 128
            att_test = torch.utils.data.Subset(attackset, odds)  # Sample
            #a = list(range(0, 40 * 128, 40))
            #attackloader = DataLoader(att_test, batch_size=64, shuffle=False)
            
        elif self.dataname == 'imagenet':
            #pass  # Add later if you wanna use it
            data_path = 'C:/Users/ChangGun Choi/Team Project/Thesis_data/val'
         
            testset = torchvision.datasets.ImageNet('./data', train=True,  download=True, transform=train_transform)
            sample = list(range(0, len(testset), args.data_divide))   # 16 * 3125 * 0.01 : 500
            valset = torch.utils.data.Subset(testset, sample) 
            val_loader = torch.utils.data.DataLoader(valset, args.batch_size ,drop_last=True)
            

        weights = self.make_weights(trainset.targets, len(trainset.classes))  # Class 확률을 가중치로 들어서 
                                                                             
        weights = torch.DoubleTensor(weights) # numpy to Tensor
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)) # 각 클래스가 나올 확률을 동일하게 해줌
        
        trainloader = DataLoader(trainset, batch_size=self.batch_size, sampler=sampler)  # use sampler 
        valloader = DataLoader(valset, batch_size=self.batch_size, shuffle=False)
        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)
        ##################
        "변경"
        "batch_size=1: Attack은 각각하니까"
        attackloader = DataLoader(att_test, batch_size=1, shuffle=False)  # batch_size=1: Attack은 각각하니까   
                                # att_test
        return trainloader, valloader, testloader, attackloader


#%% Test 용 

def imshow(img):
    plt.figure(figsize=(100,100))
    plt.imshow(img.permute(1,2,0).numpy())
    plt.savefig('pacth_example.png')
    
#%%
if __name__ == "__main__":    # Main 일때만 실행됨: 따라서 vit.py 실행할때는 __main__" 실행x 
    print("Testing Flattened2Dpatches..") 
    batch_size = 64 
    patch_size = 8   # 64/ 8 
    img_size =   32 
    num_patches = int((img_size*img_size)/(patch_size*patch_size))
                  # (32*32) / (8*8) = 4*4= 16
    # Class
    Class_Flatten = Flattened2Dpatches(dataname='cifar10', img_size=img_size, patch_size=patch_size, batch_size=batch_size)
    # Def
    trainloader, valloader, testloader, attackloader = Class_Flatten.patchdata()        #  Load patchdata 'Function'
    attackloader
    ##64*3
    # Sample 확인할떄 inter().next()사용!!
    images, labels = iter(testloader).next()   # Get Only 1 Batch  # __next__() for list() iterable
    images
    print(images.size(), labels.size())         # 1개 batch: torch.Size([64, 16, 192]) torch.Size([64])   
    "patches.reshape(batch_size, num_patch, -1)"
    
    # Flatten data를 다시 사각형 이미지로 만듬
    sample = images.reshape(batch_size, num_patches, -1, patch_size, patch_size)[0]  # -1 은 필요한것들 맞춰주고 나서 나머지
    print("Sample image size: ", sample.size())  #  torch.Size([16, 3, 8, 8])
    #imshow(torchvision.utils.make_grid(sample, nrow=int(img_size/patch_size)))  # Make_grid: Patch를 하나로 만들어줌 
    


    
    
    
    
    
    