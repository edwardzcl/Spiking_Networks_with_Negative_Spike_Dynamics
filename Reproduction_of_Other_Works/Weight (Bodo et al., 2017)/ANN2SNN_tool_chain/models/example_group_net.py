from spike_layers import *
import torch.nn as nn

class ExampleGroupNet1(nn.Module):
    def __init__(self,n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.relu1=nn.ReLU()
        self.avgpool1=nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3,groups=8)
        self.relu2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(16, 28, 3,padding=1,groups=4)
        self.relu2_2 = nn.ReLU()
        self.avgpool2=nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(28, 32, 3,groups=2)
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
