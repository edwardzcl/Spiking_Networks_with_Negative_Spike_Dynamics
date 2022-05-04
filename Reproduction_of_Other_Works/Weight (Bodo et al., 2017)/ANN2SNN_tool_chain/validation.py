import torch
import time
from spike_tensor import SpikeTensor
import spike_tensor
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate_snn(net,test_loader,device,criterion,timesteps):
    """Perform validation for SNN"""
    print("Performing validation for SNN")
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    net.eval()
    end = time.time()
    
    total_firing_ratios=[]
    with torch.no_grad():
        for data_test in test_loader:
            data, target = data_test

            noise_ratio = 0.6

            mask = (np.random.rand(data.shape[0], data.shape[1], data.shape[2], data.shape[3]) - 0.5) * noise_ratio
            mask = torch.from_numpy((1.0 + mask))
            mask = torch.tensor(mask,dtype=torch.float)


            mask = mask.to(device)
            #data = data * torch.from_numpy((1.0 + mask))
            #data = data.to(torch.double)
            #print(data.type(),data.dtype)

            #print(type(data),data.shape)

            data = data.to(device)
            data = data * mask
            replica_data = torch.cat([data for _ in range(timesteps)], 0)
            data = SpikeTensor(replica_data, timesteps, scale_factor=1)
            spike_tensor.firing_ratio_record=True
            output=net(data)
            spike_tensor.firing_ratio_record=False
            target = target.to(device)
            if isinstance(output,tuple):
                loss=0
                for o in output:
                    o=o.to_float()
                    loss+=criterion(o,target)
                output=output[0].to_float()
            else:
                # print("Debug validation.validate_snn output",output)
                output = output.to_float()
                loss = criterion(output, target)
            if len(output[0])>=5:
                topk=(1,5)
            else:
                topk=(1,1)
            prec1, prec5 = accuracy(output.data, target, topk=topk)
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

            total_firing_ratios.append([_.mean().item() for _ in spike_tensor.firing_ratios])
            spike_tensor.firing_ratios=[]
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    total_firing_ratios=np.mean(total_firing_ratios,0)
    mean_firing_ratio=total_firing_ratios.mean()
    print(f"Mean Firing ratios {mean_firing_ratio}, Firing ratios: {total_firing_ratios}")
    
    for layer in net.modules():
        if hasattr(layer,'mem_potential'):
            layer.mem_potential=None
    print(f'SNN Prec@{topk[0]} {top1.avg:.3f}, Prec@{topk[1]} {top5.avg:.3f}, Time {batch_time.sum:.5f}, Loss: {losses.avg:.3f}')
    return top1.avg, losses.avg, total_firing_ratios

def validate_ann(net,test_loader, device, criterion):
    """Perform validation for ANN"""
    print("Performing validation for ANN")
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    net.eval()
    end = time.time()
    with torch.no_grad():
        for data_test in test_loader:
            data, target = data_test
            #print(data.type(),data.dtype)
            data = data.to(device)
            output = net(data)

            target = target.to(device)
            if isinstance(output,tuple):
                loss=0
                for o in output:
                    loss+=criterion(o,target)
                output=output[0]
            else:
                loss = criterion(output, target)
            if len(output[0])>=5:
                topk=(1,5)
            else:
                topk=(1,1)
            prec1, prec5 = accuracy(output.data, target, topk=topk)
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print(f'ANN Prec@{topk[0]} {top1.avg:.3f}, Prec@{topk[1]} {top5.avg:.3f}, Time {batch_time.sum:.5f}, Loss: {losses.avg:.3f}')
    return top1.avg, losses.avg