import copy
import torch
import torch.nn as nn
from spike_layers import *

"""
global var: layer_quantization_vars
- weight_bitwidth: int
- channel_wise: bool
"""
layer_quantization_vars = {}
parameter_quantization_vars = {}


def quantize_tensor(tensor, bitwidth, channel_wise=False, rescale_back=False):
    if channel_wise:
        _max = tensor.abs().view(tensor.size(0), -1).max(1)[0]
    else:
        _max = tensor.abs().max()
    alpha = _max / (2 ** (bitwidth - 1) - 1)  # fp/int
    if channel_wise:
        if len(tensor.size()) == 4:
            alpha = alpha.view(-1, 1, 1, 1)
        elif len(tensor.size()) == 2:
            alpha = alpha.view(-1, 1)
        else:
            raise NotImplementedError
    new_tensor = torch.round(tensor / alpha)
    if rescale_back:
        new_tensor = new_tensor * alpha
    # print("Debug:quantization.new tensor",new_tensor)
    return new_tensor, alpha


def init_quantize_net(net, args):
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            # if isinstance(m,SpikeConv2d):
            if hasattr(m.weight, 'weight_float'):
                continue
            layer_quantization_vars[m] = {
                'weight_bitwidth': args.weight_bitwidth,
                'channel_wise': args.quantization_channel_wise
            }
            for p in m.parameters():
                parameter_quantization_vars[p] = {
                    'weight_bitwidth': args.weight_bitwidth,
                    'channel_wise': args.quantization_channel_wise,
                    'int_max': 2 ** (args.weight_bitwidth - 1) - 1,
                    'int_min': -2 ** (args.weight_bitwidth - 1),
                    'alpha': None,
                }
            m.weight.weight_float = m.weight.data.clone()  # ?
            # if m.bias is not None:
            #     raise NotImplementedError
        if isinstance(m, SpikeReLU):
            m.quantize = True
            m.quantize = False
            m.activation_bitwidth = args.activation_bitwidth


def quantize_layers(rescale_back=True):
    for layer, quantization_var in layer_quantization_vars.items():
        with torch.no_grad():
            # layer.weight.weight_float=layer.weight.weight_float.clamp(-1,1)
            # mean=layer.weight.weight_float.abs().view(layer.weight.size(0),-1).mean(-1).view(-1,*[1 for i in range(layer.weight.dim()-1)])
            quantized_w, alpha = quantize_tensor(layer.weight.weight_float, quantization_var['weight_bitwidth'],
                                                 quantization_var['channel_wise'], rescale_back)
            parameter_quantization_vars[layer.weight]['alpha'] = alpha
            layer.weight.data[...] = quantized_w
            layer.weight.alpha = alpha


class QuantSGD(torch.optim.SGD):
    """
    refactor torch.optim.SGD.step()
    For supporting the STE(Straight Through Estimator)
    """

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if hasattr(p, 'weight_float'):
                    weight_data = p.weight_float
                else:
                    weight_data = p.data

                if p.grad is None:
                    continue
                # STE approximate function
                d_p = p.grad.data  # ？
                # if hasattr(p,'weight_float'):
                #     d_p=d_p.clamp(parameter_quantization_vars[p]['alpha']*parameter_quantization_vars[p]['int_min'],
                #                     parameter_quantization_vars[p]['alpha']*parameter_quantization_vars[p]['int_max'])
                if weight_decay != 0:
                    # TODO: Explore the weight_decay
                    d_p.add_(weight_decay, weight_data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                weight_data.add_(-group['lr'], d_p)
        return loss


def quantize_train(epoch, net, trainloader, optimizer, criterion, device):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        quantize_layers()
        outputs = net(inputs)
        if isinstance(outputs, tuple):
            loss = 0
            for o in outputs:
                loss += criterion(o, targets)
            outputs = outputs[0]
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 60 == 59:
            print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def quantize_finetune(raw_net, trainloader, criterion, device, args):
    channel_wise = args.quantization_channel_wise
    net = copy.deepcopy(raw_net).to(device)
    optimizer = QuantSGD(net.parameters(), args.finetune_lr,
                         args.finetune_momentum, weight_decay=args.finetune_wd)
    step_epochs = [int(args.finetune_epochs * 0.5), int(args.finetune_epochs * 0.75)]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, step_epochs, 0.1)
    init_quantize_net(net, args)
    for epoch in range(0, args.finetune_epochs):
        quantize_train(epoch, net, trainloader, optimizer, criterion, device)
        lr_scheduler.step(epoch)
    quantize_layers()
    layer_quantization_vars.clear()
    return net


if __name__ == '__main__':
    # test quantization module
    from argparse import Namespace
    from torchvision.models import resnet18
    from datasets import get_dataset

    args = Namespace(dataset='ImageNet', dataset_root='./datasets',
                     finetune_lr=1e-3, finetune_epochs=2, finetune_wd=5e-5, finetune_momentum=0.9,
                     weight_bitwidth=8, quantization_channel_wise=True)
    net = resnet18(pretrained=True)
    test_loader, train_val_loader = get_dataset(args)
    qnet = quantize_finetune(net, train_val_loader, nn.CrossEntropyLoss(), torch.device('cuda'), args)
