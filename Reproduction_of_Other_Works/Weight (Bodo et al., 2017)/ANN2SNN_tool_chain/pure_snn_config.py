import torch
import torch.nn as nn
import torch.nn.functional as F

from spike_tensor import SpikeTensor
from tdlayers import tdLayer, LIFSpike

###################### settings start #####################################
####### tdLayers test example ###########
class SpikeNN(nn.Module):
    def __init__(self):
        super(SpikeNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=1, stride=1, bias = False)
        w1 = torch.ones((1, 1, 1, 1))
        self.conv1.weight = nn.Parameter(w1)
        self.conv1.weight.requires_grad = False
        
        self.conv2 = nn.Conv2d(1, 1, kernel_size = 2, stride = 2, bias = False)
        w1 = torch.ones((1, 1, 2, 2))
        self.conv2.weight = nn.Parameter(w1)
        self.conv2.weight.requires_grad = False

        self.conv1_s = tdLayer(self.conv1)
        self.conv2_s = tdLayer(self.conv2)

        self.spike = LIFSpike()
        
    def forward(self, x):
        x = self.conv1_s(x)
        x = self.spike(x)
        x = self.spike(x)
        x = self.conv2_s(x)
        x = self.spike(x)
        x = self.spike(x)
        out = torch.sum(x, dim=2) / steps  # [N, neurons, steps]
        return out

####### normal format test example ###########
# class SpikeNN(nn.Module):
#     def __init__(self):
#         super(SpikeNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 1, kernel_size=1, stride=1, bias = False)  
#         self.conv2 = nn.Conv2d(1, 1, kernel_size=2, stride=2, bias = False)


#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return x

# 设置你的输入大小
# 输入要是二值化的，不用带timestep维度 32 64 n = 1 
in_data = torch.zeros([1, 1, 8, 8]) #  n   c h w
                                    #(t*1) c h w
in_data[0][0][0][0] = 1
View = 1 #如果有View拉直层，那将View设置1

#设置每层的Vth
Vths = [torch.tensor([[1]]),
        torch.tensor([[1]]),
        torch.tensor([[1]]),
        torch.tensor([[1]]),
        torch.tensor([[1]]),
        torch.tensor([[1]]),
        torch.tensor([[1]]),
        torch.tensor([[1]]),
        torch.tensor([[1]]),
        torch.tensor([[1]]),
        torch.tensor([[1]]),
        torch.tensor([[1]]),
        torch.tensor([[1]]),
        torch.tensor([[1]])]
###################### settings end #####################################