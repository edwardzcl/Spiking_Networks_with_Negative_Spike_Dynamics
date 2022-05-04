# !/usr/bin/env python
# coding: utf-8
# Author: Chen Weiqian

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import datasets
from build_network import get_net_by_name

torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('net_name', help='network name to train')
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--dataset', default='CIFAR10', type=str, help='the location of the dataset')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for dataset')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--test_batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Preparing the dataset
print('==> Preparing data..')
train_loader, test_loader = datasets.get_dataset(args)

# Build Model
net = get_net_by_name(args.net_name)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
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
            print('train: ', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            if isinstance(outputs, tuple):
                loss = 0
                for o in outputs:
                    loss += criterion(o, targets)
                outputs = outputs[0]
            else:
                loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 60 == 59:
                print('test: ', batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc >= best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/{args.net_name}.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)
