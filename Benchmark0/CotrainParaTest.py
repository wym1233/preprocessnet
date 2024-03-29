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

def GetRDpointPerImg(img,quality,preprocessmodel):

    imghat = preprocessmodel(img)

    img=torch.squeeze(img)
    imghat=torch.squeeze(imghat)

    imghat = transforms.ToPILImage()(imghat)#PIL
    img = transforms.ToPILImage()(img)#PIL

    out,dec=JpegCompress(img,quality)
    psnr=compute_psnr(a=img,b=dec)

    outhat, dechat = JpegCompress(imghat, quality)
    psnrhat=compute_psnr(a=imghat,b=dechat)

    return [out["bpp"],psnr,outhat["bpp"],psnrhat]

def Jpeg_net_RDtest(dataloader, model, JpegQuality,logger,datadir):
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    logger.info('Testing JpegQuality:' + str(JpegQuality))
    logger.info('Testing Files length:' + str(len(dataloader)))
    assert JpegQuality>=5 and JpegQuality<=95
    npname=os.path.join(datadir,str(JpegQuality)+'_Qlt.npy')
    ls=[]
    model.eval()
    for batch_step, (images, bpp) in enumerate(dataloader):
        images=images.to(device)
        result=GetRDpointPerImg(img=images,quality=JpegQuality,preprocessmodel=model)
        ls.append(result)
    tmp = np.array(ls)
    np.save(npname, tmp)
    plotdata=tmp.mean(axis=0)
    logger.info(' result savd as '+str(npname))
    logger.info('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    return plotdata

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


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_dir = data_path
        with open(data_path, 'r') as f:
            self.lines=f.readlines()
        self.lines=self.lines[0:100]

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
                                   ckptdir=os.path.join('/data/wym123/paradata','RDVdsrParaTest5'))
    logger = getlogger(training_config.logdir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    net = Preprocess()
    state_dict_com = torch.load('/data/wym123/paradata/RDtrainPara_5/epoch_2.pth',map_location='cpu')
    net.load_state_dict(state_dict_com['model'])
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
        shuffle=False
    )
    X1=[]
    X2=[]
    Y1=[]
    Y2=[]
    for i in range(5,46,4):
        q,w,e,r=Jpeg_net_RDtest(dataloader=test_dataloader,model=net,JpegQuality=i,logger=logger,datadir=training_config.ckptdir)
        X1.append(q)
        Y1.append(w)
        X2.append(e)
        Y2.append(r)
    pltdata=[X1,Y1,X2,Y2]

    name=os.path.join(training_config.ckptdir, '../Plotdata.npy')
    tmp = np.array(pltdata)
    np.save(name, tmp)








