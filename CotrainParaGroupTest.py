from ast import parse
import os, argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
from PIL import Image
import logging
from Model import Preprocess
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
import time
import io
import pickle
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
    return out, rec
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

import PIL
def GetRDpointPerImg(img,quality):

    out,rec=JpegCompress(img,quality)
    return out["bpp"],rec

def Preprocess_Jpeg_RD(dataloader, model,Tag, Qualitylist,logger):
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    logger.info('Testing Lambda:' + str(Tag))
    logger.info('Testing Files length:' + str(len(dataloader)))
    model.eval()
    X=[]
    Y=[]
    for Q in Qualitylist:
        lsbpphat=[]
        lspsnrhat=[]
        for batch_step, (images, bpp0) in enumerate(dataloader):
            imghat = model(images)

            if type(images) != PIL.Image.Image:
                images = torch.squeeze(images)
                images = transforms.ToPILImage()(images)  # PIL
            if type(imghat) != PIL.Image.Image:
                imghat = torch.squeeze(imghat)
                imghat = transforms.ToPILImage()(imghat)  # PIL

            bpphat, dechat = GetRDpointPerImg(img=imghat, quality=Q)
            lsbpphat.append(bpphat)
            psnrhat = compute_psnr(a=images, b=dechat)
            lspsnrhat.append(psnrhat)

        X.append(sum(lsbpphat)/len(lsbpphat))
        Y.append(sum(lspsnrhat)/len(lspsnrhat))
    logger.info('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    return X,Y
def Jpeg_RD(dataloader, Qualitylist,logger):
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    logger.info('Testing JPEGRD:')
    logger.info('Testing Files length:' + str(len(dataloader)))
    X=[]
    Y=[]
    for Q in Qualitylist:
        lsbpphat=[]
        lspsnrhat=[]
        for batch_step, (images, bpp0) in enumerate(dataloader):
            imghat = images
            if type(images) != PIL.Image.Image:
                images = torch.squeeze(images)
                images = transforms.ToPILImage()(images)  # PIL
            if type(imghat) != PIL.Image.Image:
                imghat = torch.squeeze(imghat)
                imghat = transforms.ToPILImage()(imghat)  # PIL

            bpphat, dechat = GetRDpointPerImg(img=imghat, quality=Q)
            lsbpphat.append(bpphat)
            psnrhat = compute_psnr(a=images, b=dechat)
            lspsnrhat.append(psnrhat)

        X.append(sum(lsbpphat)/len(lsbpphat))
        Y.append(sum(lspsnrhat)/len(lspsnrhat))
    logger.info('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    return X,Y

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test_dataset", default='/data/wym123/paradata/bpp_25_test.txt')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=1)
    args = parser.parse_args()
    return args

class OutputConfig():
    def __init__(self, logdir, ckptdir):
        self.logdir = logdir
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
        self.lines=random.sample(self.lines,50)

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
                                   ckptdir=os.path.join('/data/wym123/paradata','RDVdsrParaTestGroup'))
    logger = getlogger(training_config.logdir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    net = Preprocess()

    paraGroup={}
    paraGroup['10']='/data/wym123/paradata/RDtrainPara/epoch_5.pth'
    paraGroup['1']='/data/wym123/paradata/RDtrainPara_1/epoch_5.pth'
    paraGroup['1e-1']='/data/wym123/paradata/RDtrainPara_2/epoch_2.pth'
    paraGroup['1e-2']='/data/wym123/paradata/RDtrainPara_3/epoch_2.pth'
    paraGroup['1e-6']='/data/wym123/paradata/RDtrainPara_4/epoch_4.pth'
    paraGroup['1e-4']='/data/wym123/paradata/RDtrainPara_5/epoch_5.pth'

    # data
    test_dataset = BaseDataset(args.test_dataset)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    plotdata={}
    Quality = list(range(1, 21, 1))
    for key in paraGroup.keys():
        state_dict_com = torch.load(paraGroup[key],map_location='cpu')
        net.load_state_dict(state_dict_com['model'])
        for (name, param) in net.named_parameters():
            param.requires_grad = False
        net=net.to(device)
        X,Y=Preprocess_Jpeg_RD(dataloader=test_dataloader,model=net,Tag=str(key),Qualitylist=Quality,
                               logger=logger)
        plotdata[key + 'X'] = X
        plotdata[key + 'Y'] = Y

    X, Y = Jpeg_RD(dataloader=test_dataloader, Qualitylist=Quality,logger=logger)
    plotdata['JPGX'] = X
    plotdata['JPGY'] = Y

    name=os.path.join(training_config.ckptdir,'plotdata_lowbpp.pkl')
    with open(name, 'wb') as f:
        pickle.dump(plotdata, f)



















