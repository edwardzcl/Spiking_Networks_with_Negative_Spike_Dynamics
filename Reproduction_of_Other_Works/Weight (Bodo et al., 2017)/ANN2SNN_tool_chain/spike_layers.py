import torch
import torch.nn as nn
import torch.nn.functional as F
from models.relu_threshold_shift import relu_thresh_shift
from spike_tensor import SpikeTensor


def wrap_one_in_one_out_func(func):
    """
    wrap for dropout
    """
    def new_func(input, *args, **kwargs):
        if isinstance(input, SpikeTensor):
            out = SpikeTensor(func(input.data, *args, **kwargs), input.timesteps, input.scale_factor)
        else:
            out = func(input, *args, **kwargs)
        return out

    return new_func


F.dropout = wrap_one_in_one_out_func(F.dropout)


def generate_spike_mem_potential(out_s, mem_potential, Vthr, reset_mode):
    """
    out_s: is a Tensor of the output of different timesteps [timesteps, *sizes]
    mem_potential: is a placeholder Tensor with [*sizes]
    spikes: [-1, *sizes]
    """
    assert reset_mode == 'subtraction'
    spikes = []
    for t in range(out_s.size(0)):
        mem_potential += out_s[t]
        spike = (mem_potential >= Vthr).float()
        mem_potential -= spike * Vthr
        spikes.append(spike)
    return spikes


class SpikeReLU(nn.Module):
    def __init__(self, threshold, quantize=False):
        super().__init__()
        self.max_val = 1
        self.quantize = quantize
        self.threshold = threshold
        self.activation_bitwidth = None

    def forward(self, x):
        if isinstance(x, SpikeTensor):
            return x
        else:
            if self.threshold is None:
                return F.relu(x)
            else:
                return relu_thresh_shift(x, self.threshold, 0)


class SpikeConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', bn=None):
        # TODO : add batchnorm here
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)
        self.mem_potential = None
        self.register_buffer('out_scales', torch.ones(out_channels))
        self.register_buffer('Vthr', torch.ones(1))
        self.register_buffer('leakage', torch.zeros(out_channels))
        self.reset_mode = 'subtraction'
        self.quant_base = None
        self.in_scales = None
        self.bn = bn
        self.spike_generator = generate_spike_mem_potential

    def forward(self, x):
        if isinstance(x, SpikeTensor):
            if self.in_scales is not None:
                inp = x.data * self.in_scales
            else:
                inp = x.data
            Vthr = self.Vthr.view(1, -1, 1, 1)
            S = F.conv2d(inp, self.weight, self.bias, self.stride, self.padding, self.dilation,
                         self.groups)
            if self.bn is not None:
                S = self.bn(S)
            chw = S.size()[1:]
            out_s = S.view(x.timesteps, -1, *chw)
            self.mem_potential = torch.zeros(out_s.size(1), *chw).to(out_s.device)
            spikes = self.spike_generator(out_s, self.mem_potential, Vthr, self.reset_mode)
            out = SpikeTensor(torch.cat(spikes, 0), x.timesteps, self.out_scales)
            # print("debug spike_layers.SpikeConv2d",out.size())
            return out
        else:
            out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            if self.bn is not None:
                out = self.bn(out)
            return out


class SpikeConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros', bn=None):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, output_padding, groups, bias, dilation, padding_mode)
        self.mem_potential = None
        self.register_buffer('out_scales', torch.ones(out_channels))
        self.register_buffer('Vthr', torch.ones(1))
        self.register_buffer('leakage', torch.zeros(out_channels))
        self.reset_mode = 'subtraction'
        self.quant_base = None
        self.spike_generator = generate_spike_mem_potential
        self.in_scales = None
        self.bn = bn

    def forward(self, x):
        if isinstance(x, SpikeTensor):
            if self.in_scales is not None:
                inp = x.data * self.in_scales
            else:
                inp = x.data
            Vthr = self.Vthr.view(1, -1, 1, 1)
            out = F.conv_transpose2d(inp, self.weight, self.bias, self.stride, self.padding, self.output_padding,
                                     self.groups, self.dilation)
            if self.bn is not None:
                out = self.bn(out)
            chw = out.size()[1:]
            out_s = out.view(x.timesteps, -1, *chw)
            self.mem_potential = torch.zeros(out_s.size(1), *chw).to(out_s.device)
            spikes = self.spike_generator(out_s, self.mem_potential, Vthr, self.reset_mode)
            out = SpikeTensor(torch.cat(spikes, 0), x.timesteps, self.out_scales)
            # print("debug spike_layers.SpikeTranspose2d",out.size())
            return out
        else:
            out = F.conv_transpose2d(x.data, self.weight, self.bias, self.stride, self.padding, self.output_padding,
                                     self.groups, self.dilation)
            if self.bn is not None:
                out = self.bn(out)
            return out


class SpikeLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, last_layer=False):
        super().__init__(in_features, out_features, bias)
        self.last_layer = last_layer
        self.register_buffer('out_scales', torch.ones(out_features))
        self.register_buffer('Vthr', torch.ones(1))
        self.register_buffer('leakage', torch.zeros(out_features))
        self.reset_mode = 'subtraction'
        self.quant_base = None
        self.spike_generator = generate_spike_mem_potential
        self.in_scales = None

    def forward(self, x):
        if isinstance(x, SpikeTensor):
            if self.in_scales is not None:
                inp = x.data * self.in_scales
            else:
                inp = x.data
            Vthr = self.Vthr.view(1, -1)
            out = F.linear(inp, self.weight, self.bias)
            chw = out.size()[1:]
            out_s = out.view(x.timesteps, -1, *chw)
            self.mem_potential = torch.zeros(out_s.size(1), *chw).to(out_s.device)
            spikes = self.spike_generator(out_s, self.mem_potential, Vthr, self.reset_mode)
            out = SpikeTensor(torch.cat(spikes, 0), x.timesteps, self.out_scales)
            return out
        else:
            out = F.linear(x, self.weight, self.bias)
            return out


class SpikeAdd(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.register_buffer('out_scales', torch.ones(out_features))
        self.register_buffer('Vthr', torch.ones(1))
        self.register_buffer('leakage', torch.zeros(out_features))
        self.register_buffer('bias', torch.zeros(out_features))
        self.weight = nn.Parameter(torch.ones([out_features, 2]))
        self.weight.alpha = 1
        self.reset_mode = 'subtraction'
        self.quant_base = None
        self.spike_generator = generate_spike_mem_potential

    def forward(self, a, b):
        if isinstance(a, SpikeTensor):
            Vthr = self.Vthr.view(1, -1, 1, 1)
            out = a.data * self.weight[:, 0].view(1, -1, 1, 1) + b.data * self.weight[:, 1].view(1, -1, 1,
                                                                                                 1) + self.bias.view(1,
                                                                                                                     -1,
                                                                                                                     1,
                                                                                                                     1)
            chw = out.size()[1:]
            out_s = out.view(a.timesteps, -1, *chw)
            self.mem_potential = torch.zeros(out_s.size(1), *chw).to(out_s.device)
            spikes = self.spike_generator(out_s, self.mem_potential, Vthr, self.reset_mode)
            out = SpikeTensor(torch.cat(spikes, 0), a.timesteps, self.out_scales)
            return out
        else:
            out = a * self.weight[:, 0].view(1, -1, 1, 1) + b * self.weight[:, 1].view(1, -1, 1, 1) + self.bias.view(1,
                                                                                                                     -1,
                                                                                                                     1,
                                                                                                                     1)
            return out
