from ast import parse
import numpy as np
import os, argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
from PIL import Image
import logging
from Model import CotrainModelDifJpg
import torch
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time
import io
def JpegCompress(img,quality):
    start = time.time()
    tmp = io.BytesIO()
    img.save(tmp, format="jpeg", quality=int(quality))
    enc_time = time.time() - start
    tmp.seek(0)
    size = tmp.getbuffer().nbytes
    start = time.time()
    rec = Image.open(tmp)
    rec.load()
    dec_time = time.time() - start
    bpp_val = float(size) * 8 / (img.size[0] * img.size[1])
    out = {
        "bpp": bpp_val,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }
    psnr=compute_psnr(a=img,b=rec)
    return bpp_val,psnr

def train(data, model,optim, logger,epoch,logdir):
    torch.cuda.empty_cache()  # 释放显存
    model.train()
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    logger.info('Start Training Epoch: ' + str(epoch))
    writer = SummaryWriter(logdir)

    optim.zero_grad()
    distortion,Rate,loss = model(data)
    distortion=torch.mean(distortion)
    Rate=torch.mean(Rate)
    loss=torch.mean(loss)
    torch.nn.utils.clip_grad_norm(parameters=model.parameters(),max_norm=4,norm_type=2)
    loss.backward()
    optim.step()

    model.eval()

    distortion, Rate, loss = model(data)

    datahat=model.vdsr(data).squeeze()
    datahat=torch.clamp(datahat,0,1)

    datahat = transforms.ToPILImage()(datahat)  # PIL
    bpp,psnr=JpegCompress(img=datahat,quality=25)


    writer.add_scalar('scalar/sumloss', loss.item(), epoch)
    writer.add_scalar('scalar/Rate', Rate.item(), (epoch))
    writer.add_scalar('scalar/bpp', bpp, (epoch))
    writer.add_scalar('scalar/psnr', psnr, (epoch))


    logger.info('epoch '+str(epoch)+' Training Done')
    logger.info('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    return (bpp,psnr)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_dataset", default='/data/wym123/paradata/bpp_25_train.txt')
    parser.add_argument("--test_dataset", default='/data/wym123/paradata/bpp_25_test.txt')
    parser.add_argument("--batch_size", type=int, default=1)  # train_batch_size
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=60)
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

import random
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_dir = data_path
        with open(data_path, 'r') as f:
            self.lines=f.readlines()
        random.seed(1234)
        self.lines = random.sample(self.lines, 50)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        line=self.lines[33]
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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def _compute_psnr(a, b, max_val: float = 255.0) -> float:
    mse = torch.mean((a - b) ** 2).item()
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr

def compute_psnr(a,b,max_val: float = 255.0):
    def _convert(x):
        if isinstance(x, Image.Image):
            x = np.asarray(x)
        x = torch.from_numpy(x.copy()).float().unsqueeze(0)
        if x.size(3) == 3:
            # (1, H, W, 3) -> (1, 3, H, W)
            x = x.permute(0, 3, 1, 2)
        return x
    a = _convert(a)
    b = _convert(b)
    psnr=_compute_psnr(a,b,max_val)
    return psnr
# 设置随机数种子

import matplotlib.pyplot as plt
if __name__ == '__main__':
    #config
    args = parse_args()
    training_config = OutputConfig(logdir=os.path.join('/output','logs'),
                                   ckptdir=os.path.join('/data/wym123/paradata/diffjpeg_cek2','oneimgtrain'))
    logger = getlogger(training_config.logdir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(1234)

    # model
    net=CotrainModelDifJpg()
    net.to(device)

    #data
    dataset =BaseDataset(args.test_dataset)
    img,_=dataset.__getitem__(1)
    img=img.unsqueeze(dim=0).to(device)

    optimizer=net.getoptimizer(args.lr)

    tim=[]
    bpp=[]
    psnr=[]
    for epoch in range(0, args.epoch):
        a,b=train(data=img,model=net,optim=optimizer,logger=logger,epoch=epoch,logdir=training_config.logdir)
        bpp.append(a)
        psnr.append(b)
        tim.append(epoch)
        # if epoch%5==0 or epoch==(args.epoch-1):
        #     net.savemodel(logger=logger,epoch=epoch,path=training_config.ckptdir)
    logger.info('jpg-----')
    img=img.cpu()
    pilimg=img.squeeze()
    pilimg=transforms.ToPILImage()(pilimg)
    BPP,PSNR=JpegCompress(img=pilimg,quality=25)
    logger.info('plt-----')
    plt.plot(tim, bpp)
    plt.plot(tim, [BPP]*len(tim))
    plt.savefig(os.path.join(training_config.ckptdir,'bpp2.png'))
    logger.info('bpp1.png')

    plt.clf()
    plt.plot(tim, psnr)
    plt.plot(tim, [PSNR]*len(tim))
    plt.savefig(os.path.join(training_config.ckptdir,'psnr2.png'))
    logger.info('psnr1.png')





