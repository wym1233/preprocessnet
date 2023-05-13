import os, argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from PIL import Image
from Model import SE_VGG as model
from torch import nn
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_dataset", default='E:/data/bitahub/vimeo_5/vimeo_train')
    parser.add_argument("--test_dataset", default='E:/data/bitahub/vimeo_5/vimeo_test')
    parser.add_argument("--batch_size", type=int, default=2)  # train_batch_size
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=1)
    args = parser.parse_args()
    return args

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
        image_path = os.path.join(Sevenimgsetpath, img_name)
        img = Image.open(image_path).convert("RGB")
        img=transforms.ToTensor()(img)
        bpp=torch.Tensor([bpp])
        return img,bpp

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train(dataloader, model,optim, epoch):
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('Start Training Epoch: ' + str(epoch))
    print('Training Files length:' + str(len(dataloader))+' batch')
    lossfunction=nn.MSELoss()
    for batch_step, (images, bpp) in enumerate(dataloader):
        optim.zero_grad()
        result= model(images)["avevalue"]
        print(result.shape)
        print(bpp)
        mse=lossfunction(bpp,result)
        print(mse)

        mse.backward()
        optim.step()
        return
    return
if __name__ == '__main__':
    #config
    args = parse_args()
    device = torch.device('cpu')
    setup_seed(1234)

    # model
    net=model()

    #data
    train_dataset =BaseDataset(args.train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    optimizer=net.getoptimizer(args.lr)

    for epoch in range(0, args.epoch):
        train(dataloader=train_dataloader,
              model=net,optim=optimizer,
              epoch=epoch)





