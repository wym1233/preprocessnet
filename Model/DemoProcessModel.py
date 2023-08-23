import torch
from torch import nn


class resblock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        lay = []
        lay.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1))
        lay.append(nn.BatchNorm2d(num_features=64))
        lay.append(nn.LeakyReLU(negative_slope=0.2))
        lay.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1))
        lay.append(nn.BatchNorm2d(num_features=64))
        self.lay = nn.Sequential(*lay)

    def forward(self, x):
        return x + self.lay(x)


class SmoothNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer1 = []
        layer1.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1,padding=3))
        layer1.append(nn.LeakyReLU(negative_slope=0.2))
        self.layer1 = nn.Sequential(*layer1)

        layer2 = []
        layer2.append(resblock())
        layer2.append(resblock())
        layer2.append(resblock())
        layer2.append(resblock())
        self.layer2 = nn.Sequential(*layer2)

        layer3 = []
        layer3.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,padding=3))
        layer3.append(nn.LeakyReLU(negative_slope=0.2))
        layer3.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,padding=1))
        layer3.append(nn.LeakyReLU(negative_slope=0.2))
        layer3.append(nn.Conv2d(in_channels=128, out_channels=3, kernel_size=7, stride=1,padding=1))
        self.layer3 = nn.Sequential(*layer3)

    def forward(self, x):
        y = self.layer1(x)
        z = y + self.layer2(y)
        return self.layer3(z)

# a=torch.rand(2,3,256,448)
# nt=SmoothNet()
# b=nt(a)
# print(a.shape)
# print(b.shape)
