from PIL import Image
from torchvision import transforms

# import torch
# import random
# class BaseDataset(torch.utils.data.Dataset):
#     def __init__(self, data_path):
#         self.data_dir = data_path
#         with open(data_path, 'r') as f:
#             self.lines=f.readlines()
#         random.seed(1234)
#         self.lines = random.sample(self.lines, 50)
#
#     def __len__(self):
#         return 1
#
#     def __getitem__(self, idx):
#         line=self.lines[10]
#         path, num = line.split(' ')
#         num = float(num)
#
#         bpp = round(num * 1000)
#
#         img = Image.open(path).convert("RGB")
#         # img=transforms.ToTensor()(img)
#         return img, bpp

# da=BaseDataset(data_path='D:/bitahubdownload/bpp_25_test.txt')
# img0=da.__getitem__(idx=1)
img_path='E:/vimeo/vimeo_test/86/im5.png'
img0=Image.open(img_path).convert("RGB")


transform=transforms.ToTensor()
img = transform(img0)

from Model import VDSR
import torch
net = VDSR()
state_dict_com = torch.load('D:/bitahubdownload/dfJPEG_sin_Cek1.pth', map_location='cpu')
net.load_state_dict(state_dict_com['model'])
for (name, param) in net.named_parameters():
    param.requires_grad = False

imghat=net(img)

imghat_pil = transforms.ToPILImage()(imghat)#PIL


img0.save('E:/tmp/1.png')
imghat_pil.save('E:/tmp/2.png')

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
    return out["bpp"],rec

def getrd(img,quality):
    bpp,rec=JpegCompress(img,quality)
    img=transform(img)
    rec=transform(rec)
    loss=torch.mean((img-rec)**2)
    print(loss,bpp)
    return
getrd(img0,25)
getrd(imghat_pil,25)




