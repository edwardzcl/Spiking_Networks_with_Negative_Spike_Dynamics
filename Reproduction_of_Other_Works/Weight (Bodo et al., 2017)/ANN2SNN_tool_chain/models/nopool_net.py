from spike_layers import *
import torch.nn as nn


class NoPoolNet2(nn.Module):
    def __init__(self,n_extra_layers=0):
        super().__init__()
        self.conv1=nn.Conv2d(3,8,kernel_size=3,stride=1,bias=False)
        self.relu1=nn.ReLU()
        self.conv2=nn.Conv2d(8,16,kernel_size=3,stride=2,bias=False)
        self.relu2=nn.ReLU()
        self.conv3=nn.Conv2d(16,32,kernel_size=3,stride=2,bias=False)
        self.relu3=nn.ReLU()
        self.fc1=nn.Linear(6*6*32,10,bias=False)

    def forward(self,x):
        out=self.relu1(self.conv1(x))
        out=self.relu2(self.conv2(out))
        out=self.relu3(self.conv3(out))
        out=out.view(-1,6*6*32)
        out=self.fc1(out)
        return out