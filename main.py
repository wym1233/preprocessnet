from Model import SE_VGG
import torch
net=SE_VGG(num_classes=500)
x=torch.rand(1,3,256,448)
y=net(x)
print(y.shape)