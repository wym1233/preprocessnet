from Model import SE_VGG
import torch

# net=SE_VGG()
x=torch.ones(2,3,9,9)
print(x)
x=x*255
print(x)
# y=net(x)
# print(y["classify_result"].shape)
# print(y["avevalue"])
# print(y["argmax"])
# torch.Size([2, 500])
# tensor([[0.2495],
#         [0.2495]], grad_fn=<ViewBackward0>)
# tensor([[159],
#         [301]])

# filename = 'D:/bitahubdownload/tmp'+ '.pth'
# torch.save({'model': net.state_dict()}, filename)