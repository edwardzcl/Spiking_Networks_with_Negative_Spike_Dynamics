'''Train CIFAR10 with PyTorch.'''
import sys
sys.path.insert(0,'..')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from spike_layers import SpikeConv2d,SpikeLinear

from datasets import get_dataset


import tensorboardX
import time
import argparse
import numpy as np
import GPUtil, os
import re
import math

np.set_printoptions(3)

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--base_lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default=None, help='resume from checkpoint')
parser.add_argument('--batch_size', default=128,type=int)
parser.add_argument('--test_batch_size', default=512,type=int)
parser.add_argument('--config_name',default='testfull2')
parser.add_argument('--actions',default='',type=str)
parser.add_argument('--epochs',default=90,type=int)
parser.add_argument('--parallel',default=False)
parser.add_argument('--weight_decay',default=1e-4,type=float)
parser.add_argument('--half',default=0,type=bool)
parser.add_argument('--scale',default=-1,type=float)
parser.add_argument('--gpu',default=-1,type=int)

args = parser.parse_args()
args.dataset='CIFAR100' if '100' in args.config_name else 'CIFAR10'

print(args)

if not args.parallel:
    if args.gpu==-1:
        min_mem_gpu = np.argmin([_.memoryUsed for _ in GPUtil.getGPUs()])
        print("selecting GPU {}".format(min_mem_gpu))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(min_mem_gpu)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
testloader,valloader,trainloader,trainvalloader=get_dataset(args)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

config_name=args.config_name

# build network

from build_network import get_net_by_name
net=get_net_by_name(config_name)

model_name=config_name
log_name=f'/{model_name}_original_e{args.epochs}'
writer=tensorboardX.SummaryWriter(f'log/{log_name}')


if device == 'cuda':
    net=net.cuda()
    if args.parallel:
        net = nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    import utils.loader
    # Load checkpoint.
    net.load_state_dict(torch.load(args.resume))
    # utils.loader.load_pretrained(net,args.resume)

class BWNSGD(torch.optim.SGD):
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
                if hasattr(p,'weight_back'):
                    weight_data=p.weight_back
                else:
                    weight_data=p.data

                if p.grad is None:
                    continue
                d_p = p.grad.data
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

class BWNAdam(torch.optim.Adam):
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p,'weight_back'):
                    weight_data=p.weight_back
                else:
                    weight_data=p.data
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], weight_data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                weight_data.addcdiv_(-step_size, exp_avg, denom)

        return loss

criterion = nn.CrossEntropyLoss()
is_warm_up=True if 'resnet110' in config_name else False
optimizer=BWNSGD(net.parameters(),args.base_lr,0.9,weight_decay=args.weight_decay)
# optimizer=BWNAdam(net.parameters(),args.base_lr,weight_decay=args.weight_decay)
# lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,1e-6)
lr_scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[int(args.epochs*0.5),int(args.epochs*0.75)],0.1)



quantized_layers = []
def init_quantize_net(net):
    for name,m in net.named_modules():
        if isinstance(m,SpikeConv2d) or isinstance(m,SpikeLinear):
        # if isinstance(m,SpikeConv2d):
            if hasattr(m.weight,'weight_back'):
                continue
            quantized_layers.append(m)
            m.weight.weight_back=m.weight.data.clone()
            if m.bias is not None:
                raise NotImplementedError

def quantize_layers():
    for layer in quantized_layers:
        with torch.no_grad():
            mean=layer.weight.weight_back.abs().view(layer.weight.size(0),-1).mean(-1).view(-1,*[1 for i in range(layer.weight.dim()-1)])
            pos_inds=(layer.weight.weight_back>0).float()
            layer.weight[...]=mean*pos_inds-mean*(1-pos_inds)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    init_quantize_net(net)
    quantize_layers()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        quantize_layers()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx%60==59:
            print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

best_acc=0
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print("save best acc",acc)
        torch.save(net.state_dict(),'checkpoint/{}_bwn.pth'.format(log_name))
        best_acc=acc

for epoch in range(0, args.epochs):
    train(epoch)
    lr_scheduler.step(epoch)
    test(epoch)
os.system('mv {} {}'.format('checkpoint/{}_bwn.pth'.format(log_name),
                            'checkpoint/{}_bwn_{:.3f}.pth'.format(log_name,best_acc)))