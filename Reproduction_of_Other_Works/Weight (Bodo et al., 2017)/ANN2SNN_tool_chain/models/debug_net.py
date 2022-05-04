import torch
import torch.nn as nn
import torch.nn.functional as F


class DebugNet0(nn.Module):
    # use dataset debug0, input shape [1,1,4,4]
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,1,1,bias=False)
        self.conv2=nn.Conv2d(1,2,2,bias=False)
        self.conv1.weight.data[...]=1.
        self.conv2.weight.data[...]=torch.arange(2*2*2).view(2,1,2,2)
    
    def forward(self,x):
        assert x.size(1)==1 and x.size(2)==4 and x.size(3)==4
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=x.view(-1,18)
        return x

class DebugNet1(nn.Module):
    def __init__(self):
        # use dataset debug0, input shape [1,1,4,4]
        super().__init__()
        self.conv1=nn.Conv2d(1,1,1,bias=False) #4x4
        self.conv2=nn.Conv2d(1,2,2,bias=False) #3x3
        self.conv3=nn.Conv2d(2,3,3,bias=False) #1x1
        self.conv1.weight.data[...]=1.
        self.conv2.weight.data[...]=torch.arange(2*2*2).view(2,1,2,2)
        self.conv3.weight.data[...]=torch.arange(3*2*3*3).view(3,2,3,3)
    
    def forward(self,x):
        assert x.size(1)==1 and x.size(2)==4 and x.size(3)==4
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=x.view(-1,3)
        return x

class DebugNet2(nn.Module):
    def __init__(self):
        # use dataset debug1, input shape [1,1,8,8]
        super().__init__()
        self.conv1=nn.Conv2d(1,2,1,bias=False) #8x8
        self.pool=nn.AvgPool2d(2,2) #4x4
        self.conv2=nn.Conv2d(2,2,2,bias=False) #3x3
        self.conv3=nn.Conv2d(2,3,3,padding=1,bias=False) #3x3
        self.conv1.weight.data[...]=1.
        self.conv2.weight.data[...]=torch.arange(2*2*2*2).view(2,2,2,2)
        self.conv3.weight.data[...]=torch.arange(3*2*3*3).view(3,2,3,3)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=x.view(-1,3*9)
        return x

class DebugNet3(nn.Module):
    def __init__(self):
        # use dataset debug2, input shape [1,1,1,1]
        super().__init__()
        self.conv1=nn.Conv2d(1,1,1,bias=False) #1x1
        self.conv2=nn.Conv2d(1,1,1,bias=False) #1x1
        self.conv1.weight.data[...]=1.
        self.conv2.weight.data[...]=1.

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=x.view(-1,1)
        return x

class DebugNet4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,1,1,bias=False)
        self.conv2=nn.Conv2d(1,2,2,bias=False)
        self.conv1.weight.data[...]=1.
        self.conv2.weight.data[...]=torch.arange(2*2*2).view(2,1,2,2)
    
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=x.view(-1,8)
        return x

class DebugNet5(nn.Module):
    def __init__(self):
        super().__init__()
        # use 1x1x1 channel input
        self.conv1=nn.Conv2d(1,1,1,bias=False)
        self.conv2=nn.Conv2d(1,1,1,bias=False)
        self.conv3=nn.Conv2d(1,1,1,bias=False)
        self.conv1.weight.data[...]=1.
        self.conv2.weight.data[...]=1.
        self.conv3.weight.data[...]=1.
    
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=x.view(-1,1)
        return x

class DebugNet6(nn.Module):
    def __init__(self):
        # use dataset debug1, input shape [1,1,8,8]
        super().__init__()
        self.conv1=nn.Conv2d(1,4,1,bias=False) #8x8
        self.pool=nn.AvgPool2d(2,2) #4x4
        self.conv2=nn.Conv2d(4,4,2,bias=False) #3x3
        self.conv3=nn.Conv2d(4,6,3,padding=1,bias=False) #3x3
        self.conv1.weight.data[...]=1.
        self.conv2.weight.data[...]=torch.arange(4*4*2*2).view(4,4,2,2)
        self.conv3.weight.data[...]=torch.arange(6*4*3*3).view(6,4,3,3)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=x.view(-1,6*9)
        return x


class DebugNet7(nn.Module):
    # use dataset debug0, input shape [1,1,4,4]
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1, bias=False)
        self.conv2 = nn.ConvTranspose2d(1, 1, 2, bias=False)
        self.conv1.weight.data[...] = 1.
        self.conv2.weight.data[...] = torch.arange(1 * 2 * 2).view(1, 1, 2, 2)

    def forward(self, x):
        assert x.size(1) == 1 and x.size(2) == 4 and x.size(3) == 4
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 50)
        return x

if __name__=='__main__':
    for i in range(8):
        net=eval(f"DebugNet{i}()")
        torch.save(net.state_dict(),f'checkpoint/debug_net{i}.pth')
    