import torch

firing_ratio_record = False
firing_ratios = []


def Poission_generate(float_tensor, timesteps):
    raise NotImplementedError
    return SpikeTensor()


class SpikeTensor():
    def __init__(self, data, timesteps, scale_factor):
        """
        wrapper for pytorch Tensor.
        data shape: [t*batch, c, h, w]
        """
        self.data = data
        self.timesteps = timesteps
        self.b = self.data.size(0) // timesteps
        self.chw = self.data.size()[1:]
        self.scale_factor = torch.ones([*self.data.size()[1:]]).to(data.device)
        if isinstance(scale_factor, torch.Tensor):
            dim = scale_factor.dim()
            # print(f"Debug: spike_tensor.SpikeTensor scale_factor {scale_factor} <dim {dim} <shape {scale_factor.shape}")
            if dim == 1:
                self.scale_factor *= scale_factor.view(-1, *([1] * (len(self.chw) - 1)))
            else:
                self.scale_factor *= scale_factor
        else:
            self.scale_factor.fill_(scale_factor)
        if firing_ratio_record:
            firing_ratios.append(self.firing_ratio())

    def firing_ratio(self):
        """
        calculate the firing ratio over the timesteps
        """
        spike = self.data.view(self.timesteps, -1, *self.chw) > 0
        firing_ratio = torch.mean(spike.float(), 0)
        return firing_ratio

    def size(self, *args):
        """
        wrapper for self.data.size()
        """
        return self.data.size(*args)

    def view(self, *args):
        """
        wrapper for self.data.view()
        args: [t*b, c, h, w]
        """
        return SpikeTensor(self.data.view(*args), self.timesteps, self.scale_factor.view(*args[1:]))

    def to_float(self):
        """
        transform the spike to floating-point number which approximate the number in ANN
        """
        assert self.scale_factor is not None
        firing_ratio = self.firing_ratio()
        scaled_float_tensor = firing_ratio * self.scale_factor.unsqueeze(0)
        return scaled_float_tensor

    def __repr__(self):
        """
        print class
        """
        return f"SpikeTensor T{self.timesteps} Shape({self.b} {self.chw}) ScaleFactor {self.scale_factor} \n{self.data.shape}"
