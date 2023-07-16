import torch
from torch import nn
a=torch.rand(2,3,256,448)
b=torch.rand(2,3,256,448)
c=nn.L1Loss()(a,b)
print(c)