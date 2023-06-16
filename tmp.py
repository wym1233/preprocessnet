# import torch
# print(torch.linspace(0.001, 1.500, 1500))
# import math
# a=0.025
# b=0.02
# l=0.05
#
# f=0.15*math.sqrt(1/b**2+1/a**2)
# print(f)

# def ll():
#     return [1,2,3,4]
# q,w,e,r=ll()
# print(q,)

# import PIL
# import torch
# from PIL import Image
# from torchvision import transforms
# img_path='E:/vimeo/vimeo_test/33/im1.png'
# img0=Image.open(img_path).convert("RGB")
# transform=transforms.ToTensor()
# img = transform(img0)
#
# print(type(img0)==PIL.Image.Image)
# print(type(img)==torch.Tensor)

# import random
# random.seed(1234)
# list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# slice = random.sample(list, 2)  #从list中随机获取5个元素，作为一个片断返回
# print (slice)
# print (list) #原有序列不会改变。

# paraGroup = {}
# paraGroup['10'] = '/data/wym123/paradata/RDtrainPara/epoch_5.pth'
# paraGroup['1'] = '/data/wym123/paradata/RDtrainPara_1/epoch_5.pth'
# paraGroup['1e-1'] = '/data/wym123/paradata/RDtrainPara_2/epoch_2.pth'
# paraGroup['1e-2'] = '/data/wym123/paradata/RDtrainPara_3/epoch_2.pth'
# paraGroup['1e-6'] = '/data/wym123/paradata/RDtrainPara_4/epoch_4.pth'
# paraGroup['1e-4'] = '/data/wym123/paradata/RDtrainPara_5/epoch_5.pth'
# for key in paraGroup.keys():
#     print(key,paraGroup[key])
#
# Quality=list(range(1,20,1))
# print(Quality)
# for Q in Quality:
#     print(Q)


# d={'1e-2X':[1,2,3,4,5],'1e-2Y':[2,5,3,4,6]}
# name='./test.npy'
#
# import pickle
#
# with open('test.pkl', 'wb') as f:
#     pickle.dump(d, f)
#
# with open('test.pkl', 'rb') as f:
#     data = pickle.load(f)

# l=list(range(1,21,5))
# print(l)

a='1e-2X'
b=a[:-1]
print(a)
print(b)
c={}
