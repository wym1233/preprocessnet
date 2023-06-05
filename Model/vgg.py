import torch
from torch import nn
import os
# from torch.cuda.amp import autocast

class fc(nn.Module):
    def __init__(self):
        super(fc, self).__init__()
        self.fc1 = nn.Linear(1500, 1, bias=False)
        self.fc1.weight.data= torch.linspace(0.001, 1.500, 1500)
        self.fc1.weight.requires_grad = False
    def forward(self, x):
        x = torch.softmax(torch.squeeze(x), dim=-1,)
        x = self.fc1(x)
        return x

class SE_VGG(nn.Module):
    def __init__(self):
        super().__init__()
        # define an empty for Conv_ReLU_MaxPool
        net = []

        # block 1
        net.append(nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 2
        net.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 3
        net.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 4
        net.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 5
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # add net into class property
        self.extract_feature = nn.Sequential(*net)

        # define an empty container for Linear operations
        classifier = []
        classifier.append(nn.Linear(in_features=57344, out_features=4096))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=4096, out_features=4096))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=4096, out_features=1500))

        # add classifier into class property
        self.classifier = nn.Sequential(*classifier)

        self.Fc=fc()


    def forward(self, x):
        feature = self.extract_feature(x)
        feature = feature.view(x.size(0), -1)
        classify_result = self.classifier(feature)

        avevalue=self.Fc(classify_result)
        avevalue = avevalue.view(-1, 1)

        argmax=torch.argmax(classify_result,dim=1).view(x.size(0),-1)

        out = {
            "classify_result": classify_result,
            "avevalue": avevalue,#float
            "argmax": argmax,#int
        }
        return out

    def getoptimizer(self,lr):
        params_lr_list = []
        for module_name in self._modules.keys():
            params_lr_list.append({"params": self._modules[module_name].parameters(), 'lr': lr})
        optimizer = torch.optim.SGD(params_lr_list, lr=lr)
        return optimizer

    def savemodel(self,logger,epoch,path):
        logger.info('Saving model...')
        filename = os.path.join(path, 'bppnet_epoch_' + str(epoch) + '.pth')
        torch.save({'model': self.state_dict()}, filename)
        logger.info('Saved as ' + str(filename))
        return