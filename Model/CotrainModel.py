from torch import nn
import torch
from math import sqrt
from Model import RCF
import os
from Model import SE_VGG
from Model import ssim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class Preprocess(nn.Module):#为适应参数文件进行封装
    def __init__(self):
        super(Preprocess, self).__init__()
        self.ae=VDSR()
        self.edge=RCF(rank=0)
    def forward(self,x):
        return self.ae(x)


class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out

class CotrainModel(nn.Module):
    def __init__(self):
        super(CotrainModel, self).__init__()
        self.vdsr = Preprocess()
        # state_dict_com = torch.load('/data/wym123/paradata/preprocess_00500000.pth')
        # self.vdsr.load_state_dict(state_dict_com['model'])

        self.vgg = SE_VGG()
        ckpt = torch.load('/data/wym123/paradata/vgg_CrossEntropy_ckpts/bppnet_epoch_25.pth')
        self.vgg.load_state_dict(ckpt['model'])
        for (name, param) in self.vgg.named_parameters():
            param.requires_grad = False

        # vdsr=Preprocess()
        # state_dict_com = torch.load('D:/bitahubdownload/preprocess_00500000.pth')
        # vdsr.load_state_dict(state_dict_com['model'])
        # logger.info('Load checkpoint for vdsr')
        #
        # vgg=SE_VGG(num_classes=500)
        # ckpt = torch.load('D:/bitahubdownload/vgg_para.pth')
        # vgg.load_state_dict(ckpt['model'])
        # for (name, param) in vgg.named_parameters():
        #     param.requires_grad = False
        # logger.info('Load checkpoint for vgg')
    def forward(self,images):
        images_hat = self.vdsr(images)
        Rate = torch.mean(self.vgg(255*images_hat)['avevalue'])
        distortion =1 - ssim(images,images_hat)
        sumloss = distortion +(1e-5)*Rate
        return distortion,Rate,sumloss

    def getoptimizer(self,lr):
        params_lr_list = []
        for module_name in self.vdsr._modules.keys():
            params_lr_list.append({"params": self.vdsr._modules[module_name].parameters(), 'lr': lr})
        optimizer = torch.optim.Adam(params_lr_list, betas=(0.9, 0.999), lr=lr)
        return optimizer

    def savemodel(self,logger,epoch,path):
        logger.info('Saving model...')
        filename = os.path.join(path, 'epoch_' + str(epoch) + '.pth')
        torch.save({'model': self.vdsr.state_dict()}, filename)
        logger.info('Saved as ' + str(filename))
        return

