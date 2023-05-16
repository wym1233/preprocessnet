import torch
from torch import nn
from Model import RCF

class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()

        self.edge = RCF()
        self.edge.load_state_dict(torch.load('D:/bitahubdownload/bsds500_pascal_model.pth'))

        hiddens = [8, 16, 32, 64]
        net=[]
        prev_channels=1
        for cur_channels in hiddens:
            net.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels,cur_channels,kernel_size=3,stride=2,padding=1),
                    nn.BatchNorm2d(cur_channels),
                    nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
            )
            prev_channels = cur_channels
        self.extract_feature = nn.Sequential(*net)

        classifier = []
        classifier.append(nn.Linear(in_features=128, out_features=32))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=32, out_features=1))
        self.classifier = nn.Sequential(*classifier)

    def forward(self,x):
        batchsize=x.size(0)
        x=self.edge(x)[5]
        feature=self.extract_feature(x).view(batchsize,-1)
        result=self.classifier(feature)
        return result
