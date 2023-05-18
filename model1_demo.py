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
net=model1(initckpt='D:/bitahubdownload/bsds500_pascal_model.pth')

optm=net.getoptimizer(lr=1e-3)
optm.zero_grad()
z=net(img)
z.backward()
optm.step()

print(z)

net.savemodel(path= 'D:/bitahubdownload',logger=None,epoch=1)

