from PIL import Image
import torch
from Model import myRCF
from torchvision import transforms

# img_path='E:/data/bitahub/vimeo_5/vimeo_train/0/im1_0.187.png'
#
# img=Image.open(img_path).convert("RGB")
# transform=transforms.ToTensor()
# img = transform(img)
# img=torch.unsqueeze(img,dim=0)
img=torch.rand(4,3,256,448)
print(img.shape)
net=myRCF()

# optm=net.getoptimizer(lr=1e-3)
# optm.zero_grad()
z=net(img)
# z.backward()
# optm.step()

print(z)

# net.savemodel(path= 'D:/bitahubdownload',logger=None,epoch=1)

