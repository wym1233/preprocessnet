import torch
from torch import nn
from Model import SSIM
a=torch.rand(2,3,256,448)
b=torch.rand(2,3,256,448)
c=nn.L1Loss()(a,b)
print(c)
ssim=SSIM()
print(ssim(a,a))
print(ssim(b,b))
print(1-ssim(a,b))




