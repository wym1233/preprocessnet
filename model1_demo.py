from PIL import Image
import torch
from Model import model1
from torchvision import transforms



img_path='E:/data/bitahub/vimeo_5/vimeo_train/0/im1_0.187.png'

img=Image.open(img_path).convert("RGB")
transform=transforms.ToTensor()
img = transform(img)
img=torch.unsqueeze(img,dim=0)

print(img.shape)
net=model1()
z=net(img)
print(z)

filename = 'D:/bitahubdownload/tmp'+ '.pth'
torch.save({'model': net.state_dict()}, filename)