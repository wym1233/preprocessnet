from torch import nn
import torch
import os

from Model import VDSR
from diffJPEG import JPEG
from Model import SE_VGG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CotrainModelDifJpg(nn.Module):
    def __init__(self):
        super(CotrainModelDifJpg, self).__init__()

        self.vdsr = VDSR()

        self.vgg = SE_VGG()
        ckpt = torch.load('/data/wym123/paradata/vgg_CrossEntropy_ckpts/bppnet_epoch_25.pth')
        self.vgg.load_state_dict(ckpt['model'])
        for (name, param) in self.vgg.named_parameters():
            param.requires_grad = False

        self.difjpeg=JPEG(height=256, width=448, quality=25)
        for (name, param) in self.difjpeg.named_parameters():
            param.requires_grad = False

    def forward(self,images):
        images_hat = self.vdsr(images)
        images_hat=torch.clamp(images_hat,0,1)

        Rate = torch.mean(self.vgg(255 * images_hat)['avevalue'])

        images_hat_=self.difjpeg(images_hat)
        distortion = torch.mean((images - images_hat_) ** 2)

        sumloss = distortion + (1e-4) * Rate
        return distortion, Rate, sumloss

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

