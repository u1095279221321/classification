# -*- coding: UTF-8 -*-

# by huang



import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import resnet152



from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision import models
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
EPOCH = 300   #遍历数据集次数
pre_epoch =0  # 定义已经遍历数据集的次数
BATCH_SIZE = 64     #批处理尺寸(batch_size)
LR = 0.1        #学习率

def loadtraindata():
    # path = r"E:\database\nuaa\data\train"
    path = "./data/train/"
    train_dataset = datasets.ImageFolder(path,
                            transform=transforms.Compose([
                           transforms.Resize((64, 64)),
                            # transforms.CenterCrop(224),
                            #transforms.RandomCrop(96),
                            # transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()])
                            )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=0)
    return train_loader, train_dataset

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
                                              shuffle=False, num_workers=0)
    return test_loader, test_dataset
trainloader, train_dataset = loadtraindata()
testloader, test_dataset =  loadtestdata()

print(test_dataset.class_to_idx)
# 模型定义-ResNet







net = resnet152(True)
net =net.to(device)



fc_features = net.fc.in_features
net.fc = nn.Linear(fc_features,3)
#state_dict=torch.load('./premodel/net_009.pth',map_location=device)
#net.load_state_dict(state_dict)



'''
model_path = './models/net_164.pth'
state_dict=torch.load(model_path,map_location=device)
net_state_keys = list(net.state_dict().keys())
for state, name_list in state_dict.items():
     if state == 'state_dict':
         for name, param in name_list.items():
             # name=name.replace('module.','')
             if name in net_state_keys:
                 print(name)
                 dst_param_shape = net.state_dict()[name].shape
                 if param.shape == dst_param_shape:
                    net.state_dict()[name].copy_(param.view(dst_param_shape))
                    net_state_keys.remove(name)
'''


net =net.to(device)




# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

# 训练
if __name__ == "__main__":
    best_acc = 90  #2 初始化best test accuracy
    print("Start Training, Resnet-152!")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")




                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/modn_%03d.pth' % ('model', epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
        print("Training Finished, TotalEPOCH=%d" % EPOCH)
