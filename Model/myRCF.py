import torch
import torch.nn as nn
import os

class myRCF(nn.Module):
    def __init__(self, pretrained=None):
        super(myRCF, self).__init__()
        self.conv1_1 = nn.Conv2d(  3,  64, 3, padding=1, dilation=1)
        self.conv1_2 = nn.Conv2d( 64,  64, 3, padding=1, dilation=1)
        self.conv2_1 = nn.Conv2d( 64, 128, 3, padding=1, dilation=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1, dilation=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1, dilation=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1, dilation=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1, dilation=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1, dilation=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=False)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=False)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=False)
        self.pool4 = nn.MaxPool2d(2, stride=1, ceil_mode=False)
        self.act = nn.ReLU(inplace=True)

        self.conv1_1_down = nn.Conv2d( 64, 21, 1)
        self.conv1_2_down = nn.Conv2d( 64, 21, 1)
        self.conv2_1_down = nn.Conv2d(128, 21, 1)
        self.conv2_2_down = nn.Conv2d(128, 21, 1)
        self.conv3_1_down = nn.Conv2d(256, 21, 1)
        self.conv3_2_down = nn.Conv2d(256, 21, 1)
        self.conv3_3_down = nn.Conv2d(256, 21, 1)
        self.conv4_1_down = nn.Conv2d(512, 21, 1)
        self.conv4_2_down = nn.Conv2d(512, 21, 1)
        self.conv4_3_down = nn.Conv2d(512, 21, 1)

        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_fuse = nn.Conv2d(5, 1, 1)

        self.cls=myclassifier()

    def forward(self, x):
        # input: B*3*256*448
        conv1_1 = self.act(self.conv1_1(x))
        conv1_2 = self.act(self.conv1_2(conv1_1))
        pool1   = self.pool1(conv1_2)
        conv2_1 = self.act(self.conv2_1(pool1))
        conv2_2 = self.act(self.conv2_2(conv2_1))
        pool2   = self.pool2(conv2_2)
        conv3_1 = self.act(self.conv3_1(pool2))
        conv3_2 = self.act(self.conv3_2(conv3_1))
        conv3_3 = self.act(self.conv3_3(conv3_2))
        pool3   = self.pool3(conv3_3)
        conv4_1 = self.act(self.conv4_1(pool3))
        conv4_2 = self.act(self.conv4_2(conv4_1))
        conv4_3 = self.act(self.conv4_3(conv4_2))
        pool4   = self.pool4(conv4_3)
        conv5_1 = self.act(self.conv5_1(pool4))
        conv5_2 = self.act(self.conv5_2(conv5_1))
        conv5_3 = self.act(self.conv5_3(conv5_2))


        conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.conv2_2_down(conv2_2)
        conv3_1_down = self.conv3_1_down(conv3_1)
        conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.conv3_3_down(conv3_3)
        conv4_1_down = self.conv4_1_down(conv4_1)
        conv4_2_down = self.conv4_2_down(conv4_2)
        conv4_3_down = self.conv4_3_down(conv4_3)
        out2 = self.score_dsn2(conv2_1_down + conv2_2_down) # 1*128*224
        out3 = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down)  # 1*64*112
        out4 = self.score_dsn4(conv4_1_down + conv4_2_down + conv4_3_down)  # 1*32*56

        out2=out2.view(-1,16,32,56)
        out3 = out3.view(-1 ,4, 32, 56)
        fuse = torch.cat((out2, out3, out4), dim=1)#([1, 21, 32, 56])
        return self.cls(fuse)
    def getoptimizer(self,lr):
        params_lr_list = []
        for module_name in self._modules.keys():
            params_lr_list.append({"params": self._modules[module_name].parameters(), 'lr': lr})
        optimizer = torch.optim.Adam(params_lr_list, betas=(0.9, 0.999), lr=lr)
        return optimizer

    def savemodel(self,logger,epoch,path):
        filename = os.path.join(path, 'myRCF_epoch_' + str(epoch) + '.pth')
        torch.save({'model': self.state_dict()}, filename)
        if logger:
            logger.info('Saved as ' + str(filename) + '.pth')
        return

class myclassifier(nn.Module):
    def __init__(self):
        super(myclassifier, self).__init__()
        net=[]
        net.append(nn.Conv2d(in_channels=21,out_channels=10,kernel_size=2,stride=2,))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=10, out_channels=5, kernel_size=2, stride=2, ))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=5, out_channels=2, kernel_size=2, stride=2, ))
        net.append(nn.ReLU())
        self.cla=nn.Sequential(*net)
        self.cls=nn.Linear(in_features=56,out_features=1)
    def forward(self,x):
        y=self.cla(x)
        y=y.view(-1,56)
        y=self.cls(y)
        return y
