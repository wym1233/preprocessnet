from ast import parse
import os, argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
from PIL import Image
import logging
from Model import SE_VGG
import torch
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
def test(dataloader, model, logger,logdir):
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    logger.info('Start Testing')
    logger.info('Testing Files length:' + str(len(dataloader))+' batch')
    writer = SummaryWriter(logdir)

    loss1=[]
    loss2=[]
    model.eval()
    for batch_step, (images, bpp) in enumerate(dataloader):
        images=images.to(device)
        bpp=bpp.to(device)
        result=model(255*images)
        argmax=result["argmax"]
        ave=result["avevalue"]

        loss_argmax=(bpp/1000).view(-1)-(argmax/1000).view(-1)
        loss_ave = (bpp / 1000).view(-1) - ave.view(-1)

        loss1.append(loss_argmax.item())
        loss2.append(loss_ave.item())
        writer.add_scalar('scalar/loss_argmax', loss_argmax.item(), (batch_step + 1 ))
        writer.add_scalar('scalar/loss_ave', loss_ave.item(), (batch_step + 1 ))
    tmp = np.array(loss1)
    np.save('/data/wym123/paradata/VggParaTest/loss_argmax.npy', tmp)
    tmp1 = np.array(loss2)
    np.save('/data/wym123/paradata/VggParaTest/loss_ave.npy', tmp1)

    ave1=np.mean(loss1)
    ave2=np.mean(loss2)
    std1=np.std(loss1)
    std2=np.std(loss2)
    logger.info('argmax:'+'ave:'+str(ave1)+'std:'+str(std1))
    logger.info('Ave:'+'ave:'+str(ave2)+'std:'+str(std2))

    logger.info(' Testing Done,'+' Batch step: ' + str(batch_step))
    logger.info('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    return

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test_dataset", default='/data/wym123/paradata/bpp_25_test.txt')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=1)
    args = parser.parse_args()
    return args

class OutputConfig():
    def __init__(self, logdir, ckptdir):
        self.logdir = logdir#
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        self.ckptdir = ckptdir
        if not os.path.exists(self.ckptdir):
            os.makedirs(self.ckptdir)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_dir = data_path
        with open(data_path, 'r') as f:
            self.lines=f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line=self.lines[idx]
        path, num = line.split(' ')
        num = float(num)

        bpp = round(num * 1000)

        img = Image.open(path).convert("RGB")
        img=transforms.ToTensor()(img)
        return img, bpp



def getlogger(logdir):
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M:%S')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


if __name__ == '__main__':
    #config
    args = parse_args()
    training_config = OutputConfig(logdir=os.path.join('/output','logs'),
                                   ckptdir=os.path.join('/data/wym123/paradata','VggParaTest'))
    logger = getlogger(training_config.logdir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    net = SE_VGG().to(device)
    ckpt = torch.load('/data/wym123/paradata/vgg_CrossEntropy_ckpts/bppnet_epoch_25.pth')
    net.load_state_dict(ckpt['model'])
    for (name, param) in net.named_parameters():
        param.requires_grad = False
    net=net.to(device)


    #data
    test_dataset =BaseDataset(args.test_dataset)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test(test_dataloader,net,logger,training_config.logdir)





