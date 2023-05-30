import torch
from torch import nn
import os
class model2(nn.Module):
    def __init__(self):
        super(model2, self).__init__()

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
        classifier.append(nn.Linear(in_features=128, out_features=64))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=64, out_features=32))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))

        classifier.append(nn.Linear(in_features=32, out_features=16))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=16, out_features=1))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))

        self.classifier = nn.Sequential(*classifier)

    def forward(self,x):
        x=self.view(-1,)
        feature=self.extract_feature(x).view(batchsize,-1)
        result=self.classifier(feature)
        return result

    def getoptimizer(self,lr):
        params_lr_list = []
        for module_name in self._modules.keys():
            params_lr_list.append({"params": self._modules[module_name].parameters(), 'lr': lr})
        optimizer = torch.optim.Adam(params_lr_list, betas=(0.9, 0.999), lr=lr)
        return optimizer

    def savemodel(self,logger,epoch,path):
        filename = os.path.join(path, 'model2_epoch_' + str(epoch) + '.pth')
        torch.save({'model': self.state_dict()}, filename)
        if logger:
            logger.info('Saved as ' + str(filename) + '.pth')
        return

