from PIL import Image
from torchvision import transforms
img_path='E:/vimeo/vimeo_test/33/im1.png'
img0=Image.open(img_path).convert("RGB")
transform=transforms.ToTensor()
img = transform(img0)
img=img.unsqueeze(dim=0)
from diffJPEG import JPEG
import io
import time
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
net = JPEG(height=256,width=448,quality=25)
for (name, param) in net.named_parameters():
    param.requires_grad = False

imghat=net(img).squeeze()
imghat = transforms.ToPILImage()(imghat)#PIL

_,imghat_=JpegCompress(img0,quality=25)

img0.save('E:/tmp/1.png')
imghat.save('E:/tmp/2.png')
imghat_.save('E:/tmp/3.png')
