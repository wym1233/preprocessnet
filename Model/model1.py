import torch
from torch import nn
from Model import RCF
import os
class model1(nn.Module):
    def __init__(self,initckpt='',requiresgrad=False):
        super(model1, self).__init__()

        self.edge = RCF()
        if initckpt:
            self.edge.load_state_dict(torch.load(initckpt))
        if requiresgrad==False:
            for (name, param) in self.edge.named_parameters():
                param.requires_grad = False


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

    def getoptimizer(self,lr):
        params_lr_list = []
        for module_name in self.extract_feature._modules.keys():
            params_lr_list.append({"params": self.extract_feature._modules[module_name].parameters(), 'lr': lr})
        for module_name in self.classifier._modules.keys():
            params_lr_list.append({"params": self.classifier._modules[module_name].parameters(), 'lr': lr})
        optimizer = torch.optim.Adam(params_lr_list, betas=(0.9, 0.999), lr=lr)
        return optimizer

    def savemodel(self,logger,epoch,path):
        filename = os.path.join(path, 'model1_epoch_' + str(epoch) + '.pth')
        torch.save({'model': self.state_dict()}, filename)
        if logger:
            logger.info('Saved as ' + str(filename) + '.pth')
        return

