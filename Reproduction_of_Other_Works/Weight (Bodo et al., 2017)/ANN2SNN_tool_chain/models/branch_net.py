from spike_layers import *
import torch.nn as nn
import torch.nn.functional as F

"""
BranchNet0: Testing branchy network conv-pool-b1/b2 b1-conv-pool-conv-fc b2-conv-pool-fc
"""


class BranchNet0(nn.Module):
    def __init__(self,n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, bias=False)
        self.relu1=nn.ReLU()
        self.avgpool1=nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2_2=nn.Conv2d(8,16,3)
        self.relu2_2 = nn.ReLU()
        self.avgpool2=nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(16, 32, 3, bias=False)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(4*4*32, 10, bias=False)
        self.fc2 = nn.Linear(6*6*16, 10, bias=False)
        

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.avgpool1(out)
        b1 = self.relu2(self.conv2(out))
        
        b1 = self.avgpool2(b1)
        b1 = self.relu3(self.conv3(b1))
        b1 = b1.view(-1, 4*4*32)
        b1 = self.fc1(b1)

        b2 = self.relu2_2(self.conv2_2(out))
        b2 = self.avgpool2(b2)
        b2=b2.view(-1,6*6*16)
        # print(b2.size())
        b2 = self.fc2(b2)
        return b1,b2