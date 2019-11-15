import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from PIL import Image
from torchvision import models
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 135  
pre_epoch = 0  
BATCH_SIZE = 128      
LR = 0.1      
def loadtestdata():
    #path = r"E:\database\nuaa\data\test"
    path = "./data/train/"
    test_dataset = datasets.ImageFolder(path,
                            transform=transforms.Compose([
                            transforms.Resize((64, 64)),
                            # transforms.CenterCrop(64),
                            #transforms.RandomCrop(96),
                            # transforms.RandomHorizontalFlip(),

                            transforms.ToTensor()])
                            )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=0)
    return test_loader, test_dataset



testloader, test_dataset =  loadtestdata()
#print(test_dataset.class_to_idx)
# net = ResNet18()
# net =net.to(device)
# state_dict=torch.load('./models/net_193.pth',map_location=device)
# net.load_state_dict(state_dict)

net = resnet152()
net =net.to(device)


fc_features = net.fc.in_features
net.fc = nn.Linear(fc_features,3)
state_dict=torch.load('./model/modn_001.pth',map_location=device)
net.load_state_dict(state_dict)

net =net.to(device)
print(device)


import cv2
import numpy as np
def onepic(path):
    import torch
    from torch.autograd import Variable
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    #print("Waiting Test!",end='')
    with torch.no_grad():
        net.eval()
        img =Image.open(path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()])
        images = transform(img)
        images = images.unsqueeze(0)
        images= images.to(device)
        outputs = net(images)
      
        _, predicted = torch.max(outputs.data, 1)
        p = predicted.to('cpu')
        print( get_key1(map1, p.numpy()))
    return get_key1(map1, p.numpy())   


def test2():
      import os
      root ='./data/train//'
      save = './error/'
      lsit = os.listdir(root)
      for i in lsit:
          leipath = root+i
          lsimg = os.listdir(leipath)
          savelei = save + i
          if not os.path.exists(savelei):
                os.mkdir(savelei)
          for j in lsimg:
             print(j)
             imgpath=leipath+'/'+j
             res = onepic(imgpath)
             if res != i:
                shutil.copy(imgpath,savelei)
def test3():
      import os
      root ='./data/test/'
    
      lsit = os.listdir(root)
      for i in lsit:
             print(i)
             imgpath= root+i
             res = onepic(imgpath)
            
def get_key1(dct, value):
   return list(filter(lambda k:dct[k] == value, dct))[0]



if __name__ == "__main__":
     onepic(path)





