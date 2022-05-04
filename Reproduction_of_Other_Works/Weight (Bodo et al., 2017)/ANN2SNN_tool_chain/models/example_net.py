from spike_layers import *
import torch.nn as nn
import torch.nn.functional as F
import torch

"""
ExampleNet0: 
ExampleNet0bias: 
ExampleNet1:
ExampleNet1bias:
ExampleNet2:
ExampleNet2bias:
ExampleNet3:
ExampleNet4: Testing bias
ExampleNet6: Testing the ConvTranspose, conv-pool-conv-deconv-conv-pool-conv-fc
ExampleNet7: Testing the concat operation
ExampleNet8: Testing batch norm
ExampleNet9: Testing addition in conv3

"""

class ExampleNet0(nn.Module):
    def __init__(self,n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, bias=False)
        self.relu1=nn.ReLU()
        self.avgpool1=nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        self.relu2 = nn.ReLU()
        self.avgpool2=nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(128, 256, 3, bias=False)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 512, 3, bias=False)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(512, 1024, 2, bias=False)
        self.relu5 = nn.ReLU()
        self.fc1 = nn.Linear(1*1*1024, 10, bias=False)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.avgpool1(out)
        out = self.relu2(self.conv2(out))
        out = self.avgpool2(out)
        out = self.relu3(self.conv3(out))
        out = self.relu4(self.conv4(out))
        out = self.relu5(self.conv5(out))
        out = out.view(-1, 1*1*1024)
        out = self.fc1(out)
        return out

class ExampleNet0bias(ExampleNet0):
    def __init__(self,n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)

class ExampleNet1(nn.Module):
    def __init__(self,n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, bias=False)
        self.relu1=nn.ReLU()
        self.avgpool1=nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(16, 28, 3,padding=1, bias=False)
        self.relu2_2 = nn.ReLU()
        self.avgpool2=nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(28, 32, 3, bias=False)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(4*4*32, 10, bias=False)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.avgpool1(out)
        out = self.relu2(self.conv2(out))
        out = self.relu2_2(self.conv2_2(out))
        out = self.avgpool2(out)
        out = self.relu3(self.conv3(out))
        out = out.view(-1, 4*4*32)
        out = self.fc1(out)
        return out

class ExampleNet1bias(ExampleNet1):
    def __init__(self,n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv2_2 = nn.Conv2d(16, 28, 3,padding=1)
        self.conv3 = nn.Conv2d(28, 32, 3)


class ExampleNet2(nn.Module):
    def __init__(self,n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, bias=False)
        self.relu1=nn.ReLU()
        self.avgpool1=nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(16, 28, 3,padding=1, bias=False)
        self.relu2_2 = nn.ReLU()
        self.avgpool2=nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(28, 28, 3, bias=False)
        self.relu3 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(28, 32, 3,padding=1, bias=False)
        self.relu3_2 = nn.ReLU()
        self.fc1 = nn.Linear(4*4*32, 10, bias=False)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.avgpool1(out)
        out = self.relu2(self.conv2(out))
        out = self.relu2_2(self.conv2_2(out))
        out = self.avgpool2(out)
        out = self.relu3(self.conv3(out))
        out = self.relu3_2(self.conv3_2(out))
        out = out.view(-1, 4*4*32)
        out = self.fc1(out)
        return out

class ExampleNet2bias(ExampleNet2):
    def __init__(self,n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv2_2 = nn.Conv2d(16, 28, 3,padding=1)
        self.conv3 = nn.Conv2d(28, 28, 3)
        self.conv3_2 = nn.Conv2d(28, 32, 3,padding=1)


class ExampleNet3(nn.Module):
    def __init__(self,n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, bias=False)
        self.relu1=nn.ReLU()
        self.avgpool1=nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(16, 28, 3, padding=1, bias=False)
        self.relu2_2 = nn.ReLU()
        self.conv2_3 = nn.Conv2d(28, 28, 3, padding=1, bias=False)
        self.relu2_3 = nn.ReLU()
        self.avgpool2=nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(28, 28, 3, bias=False)
        self.relu3 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(28, 32, 3,padding=1, bias=False)
        self.relu3_2 = nn.ReLU()
        self.fc1 = nn.Linear(4*4*32, 10, bias=False)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.avgpool1(out)
        out = self.relu2(self.conv2(out))
        out = self.relu2_2(self.conv2_2(out))
        out = self.relu2_3(self.conv2_3(out))
        out = self.avgpool2(out)
        out = self.relu3(self.conv3(out))
        out = self.relu3_2(self.conv3_2(out))
        out = out.view(-1, 4*4*32)
        out = self.fc1(out)
        return out

class ExampleNet4(nn.Module):
    def __init__(self,n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, bias=True)
        self.relu1=nn.ReLU()
        self.avgpool1=nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, bias=True)
        self.relu2 = nn.ReLU()
        self.avgpool2=nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(16, 32, 3, bias=True)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(4*4*32, 10, bias=False)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.avgpool1(out)
        out = self.relu2(self.conv2(out))
        out = self.avgpool2(out)
        out = self.relu3(self.conv3(out))
        out = out.view(-1, 4*4*32)
        out = self.fc1(out)
        return out

class ExampleNet5(nn.Module):
    def __init__(self,n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, bias=False)
        self.relu1=nn.ReLU()
        self.conv1_2 = nn.Conv2d(8, 8, 3,padding=1, bias=False)
        self.relu1_2=nn.ReLU()
        self.avgpool1=nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(16, 28, 3, padding=1, bias=False)
        self.relu2_2 = nn.ReLU()
        self.conv2_3 = nn.Conv2d(28, 28, 3, padding=1, bias=False)
        self.relu2_3 = nn.ReLU()
        self.conv2_4 = nn.Conv2d(28, 28, 3, padding=1, bias=False)
        self.relu2_4 = nn.ReLU()
        self.avgpool2=nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(28, 28, 3, bias=False)
        self.relu3 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(28, 28, 3,padding=1, bias=False)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(28, 32, 3,padding=1, bias=False)
        self.relu3_3 = nn.ReLU()
        self.fc1 = nn.Linear(4*4*32, 10, bias=False)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu1_2(self.conv1_2(out))
        out = self.avgpool1(out)
        out = self.relu2(self.conv2(out))
        out = self.relu2_2(self.conv2_2(out))
        out = self.relu2_3(self.conv2_3(out))
        out = self.relu2_4(self.conv2_4(out))
        out = self.avgpool2(out)
        out = self.relu3(self.conv3(out))
        out = self.relu3_2(self.conv3_2(out))
        out = self.relu3_3(self.conv3_3(out))
        out = out.view(-1, 4*4*32)
        out = self.fc1(out)
        return out

class ExampleNet6(nn.Module):
    def __init__(self,n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, bias=False)
        self.relu1=nn.ReLU()
        self.avgpool1=nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2_1 = nn.ConvTranspose2d(16, 16, 3, bias=False)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(16, 16, 3, bias=False)
        self.relu2_2 = nn.ReLU()
        self.avgpool2=nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(16, 32, 3, bias=False)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(4*4*32, 10, bias=False)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.avgpool1(out)
        out = self.relu2(self.conv2(out)) # torch.Size([128, 16, 12, 12])
        out = self.relu2_1(self.conv2_1(out)) # torch.Size([128, 16, 14, 14])
        out = self.relu2_2(self.conv2_2(out)) # torch.Size([128, 16, 12, 12])
        out = self.avgpool2(out)
        out = self.relu3(self.conv3(out))
        out = out.view(-1, 4*4*32)
        out = self.fc1(out)
        return out


class ExampleNet7(nn.Module):
    def __init__(self,n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, bias=False)
        self.relu1=nn.ReLU()
        self.avgpool1=nn.AvgPool2d(2)
        self.conv2_b1 = nn.Conv2d(8, 8, 3, bias=False)
        self.relu2_b1 = nn.ReLU()
        self.conv2_b2 = nn.Conv2d(8,8,3,bias=False)
        self.relu2_b2 = nn.ReLU()
        self.conv2_b2_2 = nn.Conv2d(8,8,3,padding=1,bias=False)
        self.relu2_b2_2 = nn.ReLU()
        self.avgpool2=nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(16, 32, 3, bias=False)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(4*4*32, 10, bias=False)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.avgpool1(out)
        out1 = self.relu2_b1(self.conv2_b1(out))
        out2 = self.relu2_b2(self.conv2_b2(out))
        out2 = self.relu2_b2_2(self.conv2_b2_2(out2))
        out=torch.cat([out1,out2],1)
        out = self.avgpool2(out)
        out = self.relu3(self.conv3(out))
        out = out.view(-1, 4*4*32)
        out = self.fc1(out)
        return out

class ExampleNet8(nn.Module):
    def __init__(self,n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, bias=False)
        self.bn1=nn.BatchNorm2d(8)
        self.relu1=nn.ReLU()
        self.avgpool1=nn.AvgPool2d(2)
        self.conv2_b1 = nn.Conv2d(8, 8, 3, bias=False)
        self.bn2_b1=nn.BatchNorm2d(8)
        self.relu2_b1 = nn.ReLU()
        self.conv2_b2 = nn.Conv2d(8,8,3,bias=False)
        self.bn2_b2=nn.BatchNorm2d(8)
        self.relu2_b2 = nn.ReLU()
        self.conv2_b2_2 = nn.Conv2d(8,8,3,padding=1,bias=False)
        self.bn2_b2_2=nn.BatchNorm2d(8)
        self.relu2_b2_2 = nn.ReLU()
        self.avgpool2=nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(16, 32, 3, bias=False)
        self.bn3=nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(4*4*32, 10, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.avgpool1(out)
        out1 = self.relu2_b1(self.bn2_b1(self.conv2_b1(out)))
        out2 = self.relu2_b2(self.bn2_b2(self.conv2_b2(out)))
        out2 = self.relu2_b2_2(self.bn2_b2_2(self.conv2_b2_2(out2)))
        out=torch.cat([out1,out2],1)
        out = self.avgpool2(out)
        out = self.relu3(self.bn3(self.conv3(out)))
        out = out.view(-1, 4*4*32)
        out = self.fc1(out)
        return out

class ExampleNet9(nn.Module):
    def __init__(self,n_extra_layers=0):
        """
        Test Addition in conv3
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5) # 28
        self.relu1=nn.ReLU()
        self.avgpool1=nn.AvgPool2d(2) # 14
        self.conv2 = nn.Conv2d(8, 16, 3) # 12
        self.relu2 = nn.ReLU()
        self.avgpool2=nn.AvgPool2d(2) # 6
        self.conv3 = nn.Conv2d(16, 16, 3,padding=1) # 6
        self.relu3 = nn.ReLU()
        self.avgpool3=nn.AvgPool2d(2) # 3
        self.fc1 = nn.Linear(3*3*16, 10)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.avgpool1(out)
        out = self.relu2(self.conv2(out))
        out = self.avgpool2(out)
        p = self.relu3(self.conv3(out))
        out= torch.add(p,out)
        out = self.avgpool3(out)
        out = out.view(-1, 3*3*16)
        out = self.fc1(out)
        return out

class ExampleSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 5, bias=False)
        self.relu1=nn.ReLU()
        self.avgpool1=nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(4, 6, 3, bias=False)
        self.relu2 = nn.ReLU()
        self.avgpool2=nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(6, 8, 3, bias=False)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(4*4*8, 10, bias=False)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.avgpool1(out)
        out = self.relu2(self.conv2(out))
        out = self.avgpool2(out)
        out = self.relu3(self.conv3(out))
        out = out.view(-1, 4*4*8)
        out = self.fc1(out)
        return out
