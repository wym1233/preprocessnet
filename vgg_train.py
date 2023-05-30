import os, argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
from PIL import Image
import logging
from torch.utils.tensorboard import SummaryWriter
from Model import SE_VGG as model
import math
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_dataset", default='/data/wym123/paradata/bpp_25_train.txt')
    parser.add_argument("--test_dataset", default='/data/wym123/paradata/bpp_25_test.txt')
    parser.add_argument("--batch_size", type=int, default=14)  # train_batch_size
    parser.add_argument("--lr", type=float, default=2e-4)
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
        img=255*transforms.ToTensor()(img)
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

def train(dataloader, model,optim, logger,epoch,logdir):
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    logger.info('Start Training Epoch: ' + str(epoch))
    logger.info('Training Files length:' + str(len(dataloader))+' batch')
    writer = SummaryWriter(logdir)

    lossfunction = nn.CrossEntropyLoss()
    model.train()

    for batch_step, (images, bpp) in enumerate(dataloader):
        images=images.to(device)

        bpp=bpp.view(-1)
        bpp=bpp.type(torch.LongTensor)
        bpp=bpp.to(device)

        optim.module.zero_grad()
        result= model(images)["classify_result"]
        mse=lossfunction(result,bpp)
        mse.backward()
        optim.module.step()

        absLoss=mse.item()

        writer.add_scalar('scalar/trainloss',absLoss, (batch_step + 1 + epoch * len(dataloader)))

        if (batch_step % 5000 == 0):
            logger.info(str(batch_step + 1) + 'batchsize images have been trained')
    logger.info('epoch '+str(epoch)+' Training Done,'+' Batch step: ' + str(batch_step))
    logger.info('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    return


def test(dataloader, model, logger, epoch, logdir):
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    logger.info('Start Testing Epoch: ' + str(epoch))
    logger.info('Testing Files length:' + str(len(dataloader)) + ' batch')
    writer = SummaryWriter(logdir)
    lossfunction = nn.L1Loss()
    model.eval()
    for batch_step, (images, bpp) in enumerate(dataloader):
        images = images.to(device)
        bpp = bpp.to(device)

        result = model(images)
        aveval=result["avevalue"]
        argmax=result["argmax"]

        mse1 = lossfunction(argmax, bpp).item()
        mse2 = lossfunction(aveval,bpp).item()
        writer.add_scalar('scalar/testloss_argmax', mse1, (batch_step + 1 + epoch * len(dataloader)))
        writer.add_scalar('scalar/testloss_aveval', mse2, (batch_step + 1 + epoch * len(dataloader)))
    logger.info('epoch ' + str(epoch) + ' Testing Done,' + ' Batch step: ' + str(batch_step))
    logger.info('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    return

if __name__ == '__main__':
    #config
    args = parse_args()
    training_config = OutputConfig(logdir=os.path.join('/output','logs'),
                                   ckptdir=os.path.join('/data/wym123/paradata','vgg_CrossEntropy_ckpts'))
    logger = getlogger(training_config.logdir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(1234)

    # model
    net=model()
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
    test_dataset = BaseDataset(args.test_dataset)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = net.module.getoptimizer(args.lr)
    optimizer = nn.DataParallel(optimizer, device_ids=range(torch.cuda.device_count()))

    for epoch in range(0, args.epoch):
        train(dataloader=train_dataloader,
              model=net,optim=optimizer,
              logger=logger,epoch=epoch,logdir=training_config.logdir,
              )
        if epoch%5==0:
            net.module.savemodel(logger=logger,epoch=epoch,path=training_config.ckptdir)
            test(dataloader=test_dataloader,
                 model=net,
                 logger=logger,
                 epoch=epoch,
                 logdir=training_config.logdir)






