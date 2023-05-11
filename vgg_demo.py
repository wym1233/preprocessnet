from Model import SE_VGG
import torch

net=SE_VGG(num_classes=500)
x=torch.rand(2,3,256,448)
y=net(x)
print(y["classify_result"].shape)
print(y["avevalue"])
print(y["argmax"])
# torch.Size([2, 500])
# tensor([[0.2495],
#         [0.2495]], grad_fn=<ViewBackward0>)
# tensor([[159],
#         [301]])