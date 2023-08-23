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
import PIL
def _compute_psnr(a, b, max_val: float = 255.0) -> float:
    mse = torch.mean((a - b) ** 2).item()
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr

def compute_psnr(a,b):
    if type(a)==torch.Tensor:
        if a.dim()==4:
            a = a.squeeze()
        a = transforms.ToPILImage()(a)  # PIL
    elif type(a)==Image.Image or type(a)==PIL.JpegImagePlugin.JpegImageFile:
        pass
    else:
        print('img a type error')
        print(type(a))
        # return

    if type(b)==torch.Tensor:
        if b.dim()==4:
            b = b.squeeze()
        b=transforms.ToPILImage()(b)  # PIL
    elif type(b)==Image.Image or type(b)==PIL.JpegImagePlugin.JpegImageFile:
        pass
    else:
        print('img b type error')
        print(type(b))
        # return

    def _convert(x):
        x = np.asarray(x)
        x = torch.from_numpy(x.copy()).float().unsqueeze(0)
        if x.size(3) == 3:
            # (1, H, W, 3) -> (1, 3, H, W)
            x = x.permute(0, 3, 1, 2)
        return x
    a = _convert(a)
    b = _convert(b)
    psnr=_compute_psnr(a,b)
    return float(psnr)
def JpegCompress(img,quality):
    if type(img)==torch.Tensor:
        img = img.squeeze()
        img = transforms.ToPILImage()(img)  # PIL
    elif type(img)==Image.Image or type(img)==PIL.JpegImagePlugin.JpegImageFile:
        pass
    else:
        print('img type error')
        return
    assert type(img)==Image.Image
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
    logger.info(type(rec))
    out = {
        "bpp": bpp_val,
        "rec": rec,
    }
    return out

def train(data, model,optim, logger,epoch,logdir,pretrain=False):
    model.train()
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    logger.info('Start Training Epoch: ' + str(epoch))
    writer = SummaryWriter(logdir)

    optim.zero_grad()
    distortion,Rate,loss = model(data,pretrain)
    loss=torch.mean(loss)
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),max_norm=4,norm_type=2)
    loss.backward()
    optim.step()

    model.eval()
    data_=model.vdsr(data)
    data_=torch.clamp(data_,0,1)
    JPGresult=JpegCompress(img=data_,quality=25)
    realJPGrec=JPGresult["rec"]
    diffJPGrec=model.difjpeg(data_)

    realJPGpsnr=compute_psnr(data,realJPGrec)
    diffJPGpsnr=compute_psnr(data,diffJPGrec)


    writer.add_scalar('scalar/loss', loss, epoch)
    writer.add_scalar('scalar/realJPGpsnr', realJPGpsnr, epoch)
    writer.add_scalar('scalar/diffJPGpsnr', diffJPGpsnr, epoch)
    logger.info('epoch '+str(epoch)+' Training Done')
    logger.info('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    return (JPGresult["bpp"],realJPGpsnr,diffJPGpsnr)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_dataset", default='/data/wym123/paradata/bpp_25_train.txt')
    parser.add_argument("--test_dataset", default='/data/wym123/paradata/bpp_25_test.txt')
    parser.add_argument("--batch_size", type=int, default=1)  # train_batch_size
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=100)
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
        line=self.lines[idx]
        path, num = line.split(' ')
        num = float(num)

        bpp = round(num * 1000)
        print(path)
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


# 设置随机数种子

import matplotlib.pyplot as plt
if __name__ == '__main__':
    #config
    args = parse_args()
    training_config = OutputConfig(logdir=os.path.join('/output','logs'),
                                   ckptdir=os.path.join('/data/wym123/paradata/diffjpeg_cek3','smoothnet_pretrain_cek'))
    logger = getlogger(training_config.logdir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(1234)

    # model
    net=CotrainModelDifJpg()
    net.to(device)

    #data
    dataset =BaseDataset(args.test_dataset)
    img,_=dataset.__getitem__(3)
    img=img.unsqueeze(dim=0).to(device)
    optimizer=net.getoptimizer(args.lr)

    tim=[]
    bpp=[]
    realpsnr=[]
    diffpsnr=[]
    for epoch in range(0, args.epoch):
        a,b,c=train(data=img,model=net,optim=optimizer,logger=logger,epoch=epoch,logdir=training_config.logdir,pretrain=True)
        tim.append(epoch)
        bpp.append(a)
        realpsnr.append(b)
        diffpsnr.append(c)
    # for epoch in range(0, args.epoch):
    #     a,b,c=train(data=img,model=net,optim=optimizer,logger=logger,epoch=epoch,logdir=training_config.logdir)
    #     tim.append(epoch)
    #     bpp.append(a)
    #     realpsnr.append(b)
    #     diffpsnr.append(c)
    net.savemodel(logger=logger,epoch='smoothnet_pretrain',path=training_config.ckptdir)

    logger.info('groundjpg-----')
    # img=img.cpu()
    out=JpegCompress(img=img,quality=25)
    BPP=out["bpp"]
    PSNR=compute_psnr(img,out["rec"])

    logger.info(BPP,PSNR)

    # logger.info('plt1-----')
    # plt.plot(tim, realpsnr)
    # plt.plot(tim,diffpsnr)
    # plt.plot(tim, [PSNR]*len(tim))
    # plt.savefig(os.path.join(training_config.ckptdir,'psnr_contrast3.png'))

    logger.info('plt2-----')
    plt.clf()
    ar=[i*10 for i in range(1+len(tim))]
    plt.scatter(x=[BPP]+bpp,y=[PSNR]+realpsnr,s=ar,marker='x')
    plt.annotate(text="+",xy=(BPP,PSNR),color='r')
    plt.savefig(os.path.join(training_config.ckptdir,'RDchanges.png'))

    # logger.info('plt3-----')
    # plt.clf()
    # ar = [i * 30 for i in range(1 + len(tim))]
    # plt.scatter(x=[BPP] + bpp, y=[PSNR] + diffpsnr, s=ar, marker='x')
    # plt.annotate(text="+", xy=(BPP, PSNR), color='r', )
    # plt.savefig(os.path.join(training_config.ckptdir, 'diffpsnr3.png'))





