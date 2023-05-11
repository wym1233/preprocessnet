import os, argparse
from Model import SE_VGG
from vgg_trainer import Trainer
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
from PIL import Image
import random
def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--train_dataset", default='/data/bitahub/vimeo_5/vimeo_train')#'/data/bitahub/vimeo_5/vimeo_train'
    parser.add_argument("--init_ckpt", default='')#output/epoch_5.pth
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--test_dataset", default='/data/bitahub/vimeo_5/vimeo_test')#'/data/bitahub/vimeo_5/vimeo_test'
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=20)
    # parser.add_argument("--check_time", type=float, default=10,  help='frequency for recording state (min).')
    parser.add_argument("--prefix", type=str, default='/data', help="prefix of checkpoints/logger, etc. e.g. FactorizedPrior, HyperPrior")
    # parser.add_argument("--model_name",default="factorized",help="or 'hyperprior' ")
    # parser.add_argument('--output', default='/output', help='folder to output images and model checkpoints')  # 输出结果保存路径
    args = parser.parse_args()
    return args

class TrainingConfig():
    def __init__(self, logdir, ckptdir, init_ckpt, lr):
        self.logdir = logdir
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        self.ckptdir = ckptdir
        if not os.path.exists(self.ckptdir):
            os.makedirs(self.ckptdir)
        self.init_ckpt = init_ckpt
        self.lr = lr


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, transform=None):
        self.data_dir = data_path
        self.dataset_list = [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.dataset_list)*7


    def __getitem__(self, idx):
        a=int(idx/7)
        b=int(idx%7)
        Sevenimgsetpath=os.path.join(self.data_dir, self.dataset_list[a])#idx
        Sevenimgsetlist= os.listdir(Sevenimgsetpath)
        img_name = Sevenimgsetlist[b]  # random.randint(0,6)
        try:
            bpp = float(img_name[4:9])
        except ValueError:
            bpp = float(img_name[4:8])
        bpp=torch.Tensor([int(1000*bpp)])
        image_path = os.path.join(Sevenimgsetpath, img_name)
        img = Image.open(image_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img,bpp
if __name__ == '__main__':
    # log
    args = parse_args()
    training_config = TrainingConfig(
                            logdir=os.path.join('/output','logs'),
                            ckptdir=os.path.join('/output','ckpts'),
                            init_ckpt=args.init_ckpt,
                            lr=args.lr, )
    # model
    model=SE_VGG(num_classes=500)

    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(config=training_config, model=model,criterion=criterion)

    train_dataset =BaseDataset(args.train_dataset,transform=transforms.ToTensor())
    test_dataset = BaseDataset(args.test_dataset,transform=transforms.ToTensor())
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )
    # print(len(train_dataset),len(test_dataset))
    # print(len(train_dataloader),len(test_dataloader))

    # training
    for epoch in range(0, args.epoch):
        trainer.train(train_dataloader)
        trainer.test(test_dataloader)
        trainer.epoch+=1

