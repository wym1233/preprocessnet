from PIL import Image
from torchvision import transforms
img_path='E:/vimeo/vimeo_test/33/im1.png'
img0=Image.open(img_path).convert("RGB")
transform=transforms.ToTensor()
img = transform(img0)

from Model import Preprocess
import torch
net = Preprocess()
state_dict_com = torch.load('D:/bitahubdownload/SSIM_cek.pth', map_location='cpu')
net.load_state_dict(state_dict_com['model'])
for (name, param) in net.named_parameters():
    param.requires_grad = False

imghat=net(img)
imghat_=torch.clamp(imghat,0,1)
imghat = transforms.ToPILImage()(imghat)#PIL
imghat_ = transforms.ToPILImage()(imghat_)#PIL
img0.save('E:/tmp/1.png')
imghat.save('E:/tmp/2.png')
imghat_.save('E:/tmp/3.png')
