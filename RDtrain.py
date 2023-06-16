from ast import parse
import os, argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
from PIL import Image
import logging
from Model import CotrainModel
import torch
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(dataloader, model,optim, logger,epoch,logdir):
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    logger.info('Start Training Epoch: ' + str(epoch))
    logger.info('Training Files length:' + str(len(dataloader))+' batch')
    writer = SummaryWriter(logdir)

    for batch_step, (images, bpp) in enumerate(dataloader):
        optim.module.zero_grad()
        distortion,Rate,loss = model(images)

        distortion=torch.mean(distortion)
        Rate=torch.mean(Rate)
        loss=torch.mean(loss)

        loss.backward()
        optim.module.step()

        bppdistortion=torch.mean(bpp/1000)-Rate

        writer.add_scalar('scalar/Distortion', distortion.item(), (batch_step + 1 + epoch * len(dataloader)))
        writer.add_scalar('scalar/Rate', Rate.item(), (batch_step + 1 + epoch * len(dataloader)))
        writer.add_scalar('scalar/sumloss', loss.item(), (batch_step + 1 + epoch * len(dataloader)))
        writer.add_scalar('scalar/bppdistortion', bppdistortion.item(), (batch_step + 1 + epoch * len(dataloader)))

        if (batch_step % 5000 == 0):
            logger.info(str(batch_step + 1) + 'batchsize images have been trained')

    logger.info('epoch '+str(epoch)+' Training Done,'+' Batch step: ' + str(batch_step))
    logger.info('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    return

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_dataset", default='/data/wym123/paradata/bpp_25_train.txt')
    parser.add_argument("--test_dataset", default='/data/wym123/paradata/bpp_25_test.txt')
    parser.add_argument("--batch_size", type=int, default=20)  # train_batch_size
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=30)
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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子


if __name__ == '__main__':
    #config
    args = parse_args()
    training_config = OutputConfig(logdir=os.path.join('/output','logs'),
                                   ckptdir=os.path.join('/data/wym123/paradata','RDtrainPara_6'))
    logger = getlogger(training_config.logdir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    setup_seed(1234)

    # model
    net=CotrainModel()
    if device == 'cuda':
        net = net.cuda()
        torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        net.to(device)

    #data
    train_dataset =BaseDataset(args.train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer=net.module.getoptimizer(args.lr)
    optimizer = nn.DataParallel(optimizer, device_ids=range(torch.cuda.device_count()))

    for epoch in range(0, args.epoch):
        train(dataloader=train_dataloader,
              model=net,optim=optimizer,
              logger=logger,epoch=epoch,logdir=training_config.logdir,
              )
        if epoch>=1:
            net.module.savemodel(logger=logger,epoch=epoch,path=training_config.ckptdir)





