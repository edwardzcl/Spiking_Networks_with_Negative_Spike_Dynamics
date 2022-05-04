import torch
import torch.nn as nn
from spike_tensor import SpikeTensor


def relu_thresh_shift(x, threshold, shift):
    y = x.clamp(0, threshold)
    return y


class ReLUThresholdShift(nn.Module):
    def __init__(self, threshold, shift_mode, timesteps):
        super().__init__()
        self.threshold = threshold
        self.shift_mode = shift_mode
        self.timesteps = timesteps
        self.shift_value = torch.zeros(1)

    def forward(self, x):
        if isinstance(x, SpikeTensor):
            return x
        else:
            # if self.training:
            #     # print("training relu threshold shift")
            #     if self.shift_mode=='V/2T':
            #         with torch.no_grad():
            #             self.shift_value=max(self.shift_value.to(x.device),relu_thresh_shift(x,self.threshold,0).max()/(2*self.timesteps))
            #     elif self.shift_mode=='0':
            #         pass
            #     else:
            #         raise NotImplementedError()
            # return relu_thresh_shift(x,self.threshold,self.shift_value.to(x.device))
            return relu_thresh_shift(x, self.threshold, 0)
