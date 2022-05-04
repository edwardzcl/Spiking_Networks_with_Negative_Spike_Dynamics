# !/usr/bin/env python
# coding: utf-8
# Author: Chen Weiqian

import torch
import torch.nn as nn
import torch.nn.functional as F


class TestNet0(nn.Module):
    def __init__(self, n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1, groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(256, 1024, 3, padding=1, groups=16, bias=False)
        self.bn3 = nn.BatchNorm2d(1024)
        self.relu3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool2d(2)
        self.conv4 = nn.Conv2d(1024, 256, 1, groups=4, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(256, 16, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(16)
        self.relu5 = nn.ReLU()
        self.fc1 = nn.Linear(4 * 4 * 16, 10, bias=False)
        self.conv6 = nn.Conv2d(256, 128, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.relu6 = nn.ReLU()
        self.tconv7 = nn.ConvTranspose2d(128, 128, 3, padding=1, stride=2, output_padding=1, groups=8, bias=False)
        self.bn7 = nn.BatchNorm2d(128)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(384, 16, 1, groups=2, bias=False)
        self.bn8 = nn.BatchNorm2d(16)
        self.relu8 = nn.ReLU()
        self.fc2 = nn.Linear(8 * 8 * 16, 10, bias=False)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.avgpool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.avgpool2(x)

        y1 = self.relu3(self.bn3(self.conv3(x)))
        y1 = self.avgpool3(y1)
        y1 = self.relu4(self.bn4(self.conv4(y1)))
        y1_1 = y1
        y1 = self.relu5(self.bn5(self.conv5(y1)))
        y1 = y1.view(-1, 4 * 4 * 16)
        y1 = self.fc1(y1)

        y1_1 = self.relu6(self.bn6(self.conv6(y1_1)))
        y1_1 = self.relu7(self.bn7(self.tconv7(y1_1)))
        y2 = torch.cat([x, y1_1], 1)
        y2 = self.relu8(self.bn8(self.conv8(y2)))
        y2 = y2.view(-1, 8 * 8 * 16)
        y2 = self.fc2(y2)

        return y1, y2


class TestNet1(nn.Module):
    """
    without BN
    """
    def __init__(self, n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1, groups=8)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(256, 1024, 3, padding=1, groups=16)
        self.relu3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool2d(2)
        self.conv4 = nn.Conv2d(1024, 256, 1, groups=4)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(256, 16, 1)
        self.relu5 = nn.ReLU()
        self.fc1 = nn.Linear(4 * 4 * 16, 10)
        self.conv6 = nn.Conv2d(256, 128, 1)
        self.relu6 = nn.ReLU()
        self.tconv7 = nn.ConvTranspose2d(128, 128, 3, padding=1, stride=2, output_padding=1, groups=8)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(384, 16, 1, groups=2)
        self.relu8 = nn.ReLU()
        self.fc2 = nn.Linear(8 * 8 * 16, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.avgpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.avgpool2(x)

        y1 = self.relu3(self.conv3(x))
        y1 = self.avgpool3(y1)
        y1 = self.relu4(self.conv4(y1))
        y1_1 = y1
        y1 = self.relu5(self.conv5(y1))
        y1 = y1.view(-1, 4 * 4 * 16)
        y1 = self.fc1(y1)

        y1_1 = self.relu6(self.conv6(y1_1))
        y1_1 = self.relu7(self.tconv7(y1_1))
        y2 = torch.cat([x, y1_1], 1)
        y2 = self.relu8(self.conv8(y2))
        y2 = y2.view(-1, 8 * 8 * 16)
        y2 = self.fc2(y2)

        return y1, y2


class TestNet2(nn.Module):
    """
    small, without BN
    """
    def __init__(self, n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1, groups=4)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(8, 32, 3, padding=1, groups=8)
        self.relu3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool2d(2)
        self.conv4 = nn.Conv2d(32, 8, 1, groups=4)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(8, 16, 1)
        self.relu5 = nn.ReLU()
        self.fc1 = nn.Linear(4 * 4 * 16, 10)
        self.conv6 = nn.Conv2d(8, 4, 1)
        self.relu6 = nn.ReLU()
        self.tconv7 = nn.ConvTranspose2d(4, 4, 3, padding=1, stride=2, output_padding=1, groups=4)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(12, 16, 1, groups=2)
        self.relu8 = nn.ReLU()
        self.fc2 = nn.Linear(8 * 8 * 16, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.avgpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.avgpool2(x)

        y1 = self.relu3(self.conv3(x))
        y1 = self.avgpool3(y1)
        y1 = self.relu4(self.conv4(y1))
        y1_1 = y1
        y1 = self.relu5(self.conv5(y1))
        y1 = y1.view(-1, 4 * 4 * 16)
        y1 = self.fc1(y1)

        y1_1 = self.relu6(self.conv6(y1_1))
        y1_1 = self.relu7(self.tconv7(y1_1))
        y2 = torch.cat([x, y1_1], 1)
        y2 = self.relu8(self.conv8(y2))
        y2 = y2.view(-1, 8 * 8 * 16)
        y2 = self.fc2(y2)

        return y1, y2


class TestNet3(nn.Module):
    """
    fc2 branch
    """
    def __init__(self, n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1, groups=4)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(8, 32, 3, padding=1, groups=8)
        self.relu3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool2d(2)
        self.conv4 = nn.Conv2d(32, 8, 1, groups=4)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(8, 16, 1)
        self.relu5 = nn.ReLU()
        self.fc1 = nn.Linear(4 * 4 * 16, 10)
        self.conv6 = nn.Conv2d(8, 4, 1)
        self.relu6 = nn.ReLU()
        self.tconv7 = nn.ConvTranspose2d(4, 4, 3, padding=1, stride=2, output_padding=1, groups=4)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(12, 16, 1, groups=2)
        self.relu8 = nn.ReLU()
        self.fc2 = nn.Linear(8 * 8 * 16, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.avgpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.avgpool2(x)

        y1 = self.relu3(self.conv3(x))
        y1 = self.avgpool3(y1)
        y1 = self.relu4(self.conv4(y1))
        y1_1 = y1
        #y1 = self.relu5(self.conv5(y1))
        #y1 = y1.view(-1, 4 * 4 * 16)
        #y1 = self.fc1(y1)

        y1_1 = self.relu6(self.conv6(y1_1))
        y1_1 = self.relu7(self.tconv7(y1_1))
        y2 = torch.cat([x, y1_1], 1)
        y2 = self.relu8(self.conv8(y2))
        y2 = y2.view(-1, 8 * 8 * 16)
        y2 = self.fc2(y2)

        return y2


class TestNet4(nn.Module):
    """
    fc1 branch
    """
    def __init__(self, n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1, groups=4)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(8, 32, 3, padding=1, groups=8)
        self.relu3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool2d(2)
        self.conv4 = nn.Conv2d(32, 8, 1, groups=4)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(8, 16, 1)
        self.relu5 = nn.ReLU()
        self.fc1 = nn.Linear(4 * 4 * 16, 10)
        #self.conv6 = nn.Conv2d(8, 4, 1)
        #self.relu6 = nn.ReLU()
        #self.tconv7 = nn.ConvTranspose2d(4, 4, 3, padding=1, stride=2, output_padding=1, groups=4)
        #self.relu7 = nn.ReLU()
        #self.conv8 = nn.Conv2d(12, 16, 1, groups=2)
        #self.relu8 = nn.ReLU()
        #self.fc2 = nn.Linear(8 * 8 * 16, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.avgpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.avgpool2(x)

        y1 = self.relu3(self.conv3(x))
        y1 = self.avgpool3(y1)
        y1 = self.relu4(self.conv4(y1))
        y1_1 = y1
        y1 = self.relu5(self.conv5(y1))
        y1 = y1.view(-1, 4 * 4 * 16)
        y1 = self.fc1(y1)

        #y1_1 = self.relu6(self.conv6(y1_1))
        #y1_1 = self.relu7(self.tconv7(y1_1))
        #y2 = torch.cat([x, y1_1], 1)
        #y2 = self.relu8(self.conv8(y2))
        #y2 = y2.view(-1, 8 * 8 * 16)
        #y2 = self.fc2(y2)

        return y1


class TestNet5(nn.Module):
    """
    without BN and bias
    """
    def __init__(self, n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1, groups=8, bias=False)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(256, 1024, 3, padding=1, groups=16, bias=False)
        self.relu3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool2d(2)
        self.conv4 = nn.Conv2d(1024, 256, 1, groups=4, bias=False)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(256, 16, 1, bias=False)
        self.relu5 = nn.ReLU()
        self.fc1 = nn.Linear(4 * 4 * 16, 10, bias=False)
        self.conv6 = nn.Conv2d(256, 128, 1, bias=False)
        self.relu6 = nn.ReLU()
        self.tconv7 = nn.ConvTranspose2d(128, 128, 3, padding=1, stride=2, output_padding=1, groups=8, bias=False)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(384, 16, 1, groups=2, bias=False)
        self.relu8 = nn.ReLU()
        self.fc2 = nn.Linear(8 * 8 * 16, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.avgpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.avgpool2(x)

        y1 = self.relu3(self.conv3(x))
        y1 = self.avgpool3(y1)
        y1 = self.relu4(self.conv4(y1))
        y1_1 = y1
        y1 = self.relu5(self.conv5(y1))
        y1 = y1.view(-1, 4 * 4 * 16)
        y1 = self.fc1(y1)

        y1_1 = self.relu6(self.conv6(y1_1))
        y1_1 = self.relu7(self.tconv7(y1_1))
        y2 = torch.cat([x, y1_1], 1)
        y2 = self.relu8(self.conv8(y2))
        y2 = y2.view(-1, 8 * 8 * 16)
        y2 = self.fc2(y2)

        return y1, y2


class TestNet6(nn.Module):
    """
    small, without BN and bias
    """
    def __init__(self, n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1, groups=4, bias=False)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(8, 32, 3, padding=1, groups=8, bias=False)
        self.relu3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool2d(2)
        self.conv4 = nn.Conv2d(32, 8, 1, groups=4, bias=False)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(8, 16, 1, bias=False)
        self.relu5 = nn.ReLU()
        self.fc1 = nn.Linear(4 * 4 * 16, 10, bias=False)
        self.conv6 = nn.Conv2d(8, 4, 1, bias=False)
        self.relu6 = nn.ReLU()
        self.tconv7 = nn.ConvTranspose2d(4, 4, 3, padding=1, stride=2, output_padding=1, groups=4, bias=False)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(12, 16, 1, groups=2, bias=False)
        self.relu8 = nn.ReLU()
        self.fc2 = nn.Linear(8 * 8 * 16, 10, bias=False)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.avgpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.avgpool2(x)

        y1 = self.relu3(self.conv3(x))
        y1 = self.avgpool3(y1)
        y1 = self.relu4(self.conv4(y1))
        y1_1 = y1
        y1 = self.relu5(self.conv5(y1))
        y1 = y1.view(-1, 4 * 4 * 16)
        y1 = self.fc1(y1)

        y1_1 = self.relu6(self.conv6(y1_1))
        y1_1 = self.relu7(self.tconv7(y1_1))
        y2 = torch.cat([x, y1_1], 1)
        y2 = self.relu8(self.conv8(y2))
        y2 = y2.view(-1, 8 * 8 * 16)
        y2 = self.fc2(y2)

        return y1, y2


class TestNet7(nn.Module):
    """
    larger than small, without BN and bias
    """
    def __init__(self, n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, groups=4, bias=False)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(32, 128, 3, padding=1, groups=8, bias=False)
        self.relu3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool2d(2)
        self.conv4 = nn.Conv2d(128, 32, 1, groups=4, bias=False)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(32, 64, 1, bias=False)
        self.relu5 = nn.ReLU()
        self.fc1 = nn.Linear(4 * 4 * 64, 10, bias=False)
        self.conv6 = nn.Conv2d(32, 16, 1, bias=False)
        self.relu6 = nn.ReLU()
        self.tconv7 = nn.ConvTranspose2d(16, 16, 3, padding=1, stride=2, output_padding=1, groups=4, bias=False)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(48, 64, 1, groups=2, bias=False)
        self.relu8 = nn.ReLU()
        self.fc2 = nn.Linear(8 * 8 * 64, 10, bias=False)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.avgpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.avgpool2(x)

        y1 = self.relu3(self.conv3(x))
        y1 = self.avgpool3(y1)
        y1 = self.relu4(self.conv4(y1))
        y1_1 = y1
        y1 = self.relu5(self.conv5(y1))
        y1 = y1.view(-1, 4 * 4 * 64)
        y1 = self.fc1(y1)

        y1_1 = self.relu6(self.conv6(y1_1))
        y1_1 = self.relu7(self.tconv7(y1_1))
        y2 = torch.cat([x, y1_1], 1)
        y2 = self.relu8(self.conv8(y2))
        y2 = y2.view(-1, 8 * 8 * 64)
        y2 = self.fc2(y2)

        return y1, y2


class TestNet8(nn.Module):
    """
    larger than small, with bias, without BN
    """
    def __init__(self, n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, groups=4)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(32, 128, 3, padding=1, groups=8)
        self.relu3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool2d(2)
        self.conv4 = nn.Conv2d(128, 32, 1, groups=4)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(32, 64, 1)
        self.relu5 = nn.ReLU()
        self.fc1 = nn.Linear(4 * 4 * 64, 10)
        self.conv6 = nn.Conv2d(32, 16, 1)
        self.relu6 = nn.ReLU()
        self.tconv7 = nn.ConvTranspose2d(16, 16, 3, padding=1, stride=2, output_padding=1, groups=4)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(48, 64, 1, groups=2)
        self.relu8 = nn.ReLU()
        self.fc2 = nn.Linear(8 * 8 * 64, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.avgpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.avgpool2(x)

        y1 = self.relu3(self.conv3(x))
        y1 = self.avgpool3(y1)
        y1 = self.relu4(self.conv4(y1))
        y1_1 = y1
        y1 = self.relu5(self.conv5(y1))
        y1 = y1.view(-1, 4 * 4 * 64)
        y1 = self.fc1(y1)

        y1_1 = self.relu6(self.conv6(y1_1))
        y1_1 = self.relu7(self.tconv7(y1_1))
        y2 = torch.cat([x, y1_1], 1)
        y2 = self.relu8(self.conv8(y2))
        y2 = y2.view(-1, 8 * 8 * 64)
        y2 = self.fc2(y2)

        return y1, y2


class TestNet9(nn.Module):
    """
    for 3 yuan
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool2d(2)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=1, groups=2)
        self.relu4 = nn.ReLU()
        self.avgpool4 = nn.AvgPool2d(2)
        self.conv5 = nn.Conv2d(64, 128, 3, stride=1, padding=1, groups=4)
        self.relu5 = nn.ReLU()
        self.avgpool5 = nn.AvgPool2d(2)
        self.conv6 = nn.Conv2d(128, 128, 3, stride=1, padding=1, groups=8)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(128, 256, 3, stride=1, padding=1, groups=8)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(256, 128, 1, stride=1, groups=16)
        self.relu8 = nn.ReLU()
        self.conv9 = nn.Conv2d(128, 72, 3, stride=1, padding=1, groups=8)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(72, 18, 1, stride=1, groups=3)
        self.relu10 = nn.ReLU()
        self.conv11 = nn.Conv2d(128, 64, 1, stride=1, groups=8)
        self.relu11 = nn.ReLU()
        self.conv_transpose2d1 = nn.ConvTranspose2d(64, 64, 3, padding=1, stride=2, output_padding=1, groups=4)
        self.relu12 = nn.ReLU()
        self.conv12 = nn.Conv2d(192, 108, 3, stride=1, padding=1, groups=12)
        self.relu13 = nn.ReLU()
        self.conv13 = nn.Conv2d(108, 18, 1, stride=1, groups=6)
        self.relu14 = nn.ReLU()

    def forward(self, x):
        y1 = self.relu1(self.conv1(x))
        a1 = self.avgpool1(y1)
        y2 = self.relu2(self.conv2(a1))
        a2 = self.avgpool2(y2)
        y3 = self.relu3(self.conv3(a2))
        a3 = self.avgpool3(y3)
        y4 = self.relu4(self.conv4(a3))
        a4 = self.avgpool4(y4)
        y5 = self.relu5(self.conv5(a4))
        a5 = self.avgpool5(y5)
        y6 = self.relu6(self.conv6(a5))
        y7 = self.relu7(self.conv7(y6))
        y8 = self.relu8(self.conv8(y7))
        y9 = self.relu9(self.conv9(y8))
        y10 = self.relu10(self.conv10(y9))
        y11 = self.relu11(self.conv11(y8))
        y12 = self.relu12(self.conv_transpose2d1(y11))
        c1 = torch.cat([y5, y12], 1)
        y13 = self.relu13(self.conv12(c1))
        y14 = self.relu14(self.conv13(y13))
        y10 = y10.view(-1, 18*4*4)
        y14 = y14.view(-1, 18*8*8)

        return y10, y14


class TestNet10(nn.Module):
    """
    for 3 yuan small
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, 3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(2, 4, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(4, 8, 3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool2d(2)
        self.conv4 = nn.Conv2d(8, 16, 3, stride=1, padding=1, groups=1)
        self.relu4 = nn.ReLU()
        self.avgpool4 = nn.AvgPool2d(2)
        self.conv5 = nn.Conv2d(16, 32, 3, stride=1, padding=1, groups=1)
        self.relu5 = nn.ReLU()
        self.avgpool5 = nn.AvgPool2d(2)
        self.conv6 = nn.Conv2d(32, 32, 3, stride=1, padding=1, groups=2)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(32, 64, 3, stride=1, padding=1, groups=2)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(64, 32, 1, stride=1, groups=1)
        self.relu8 = nn.ReLU()
        self.conv9 = nn.Conv2d(32, 18, 3, stride=1, padding=1, groups=2)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(18, 18, 1, stride=1, groups=1)
        self.relu10 = nn.ReLU()
        self.conv11 = nn.Conv2d(32, 16, 1, stride=1, groups=1)
        self.relu11 = nn.ReLU()
        self.conv_transpose2d1 = nn.ConvTranspose2d(16, 16, 3, padding=1, stride=2, output_padding=1, groups=1)
        self.relu12 = nn.ReLU()
        self.conv12 = nn.Conv2d(48, 27, 3, stride=1, padding=1, groups=3)
        self.relu13 = nn.ReLU()
        self.conv13 = nn.Conv2d(27, 18, 1, stride=1, groups=1)
        self.relu14 = nn.ReLU()

    def forward(self, x):
        y1 = self.relu1(self.conv1(x))
        a1 = self.avgpool1(y1)
        y2 = self.relu2(self.conv2(a1))
        a2 = self.avgpool2(y2)
        y3 = self.relu3(self.conv3(a2))
        a3 = self.avgpool3(y3)
        y4 = self.relu4(self.conv4(a3))
        a4 = self.avgpool4(y4)
        y5 = self.relu5(self.conv5(a4))
        a5 = self.avgpool5(y5)
        y6 = self.relu6(self.conv6(a5))
        y7 = self.relu7(self.conv7(y6))
        y8 = self.relu8(self.conv8(y7))
        y9 = self.relu9(self.conv9(y8))
        y10 = self.relu10(self.conv10(y9))
        y11 = self.relu11(self.conv11(y8))
        y12 = self.relu12(self.conv_transpose2d1(y11))
        c1 = torch.cat([y5, y12], 1)
        y13 = self.relu13(self.conv12(c1))
        y14 = self.relu14(self.conv13(y13))
        y10 = y10.view(-1, 18*4*4)
        y14 = y14.view(-1, 18*8*8)

        return y10, y14


class TestNet11(nn.Module):
    """
    for 206 uav
    """
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=2)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=1, groups=2)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(64, 64, 3, stride=2, padding=1, groups=4)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(64, 128, 3, stride=1, padding=1, groups=4)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(128, 128, 3, stride=2, padding=1, groups=8)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(128, 256, 3, stride=1, padding=1, groups=8)
        self.relu8 = nn.ReLU()
        self.conv9 = nn.Conv2d(256, 256, 3, stride=2, padding=1, groups=16)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(256, 512, 3, stride=1, padding=1, groups=16)
        self.relu10 = nn.ReLU()
        self.conv11 = nn.Conv2d(512, 1024, 3, stride=1, padding=1, groups=32)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(1024, 256, 1, stride=1, padding=0, groups=4)
        self.relu12 = nn.ReLU()
        self.conv13 = nn.Conv2d(256, 512, 3, stride=1, padding=1, groups=16)
        self.relu13 = nn.ReLU()

        self.conv17 = nn.Conv2d(256, 128, 1, stride=1, padding=0, groups=1)
        self.relu17 = nn.ReLU()
        self.conv_transpose2d18 = nn.ConvTranspose2d(128, 128, 3, padding=1, stride=2, output_padding=1, groups=8)
        self.relu18 = nn.ReLU()
        self.conv20 = nn.Conv2d(384, 256, 3, stride=1, padding=1, groups=16)
        self.relu20 = nn.ReLU()

    def forward(self, x):
        y0 = self.relu0(self.conv0(x))
        y1 = self.relu1(self.conv1(y0))
        y2 = self.relu2(self.conv2(y1))
        y3 = self.relu3(self.conv3(y2))
        y4 = self.relu4(self.conv4(y3))
        y5 = self.relu5(self.conv5(y4))
        y6 = self.relu6(self.conv6(y5))
        y7 = self.relu7(self.conv7(y6))
        y8 = self.relu8(self.conv8(y7))
        y9 = self.relu9(self.conv9(y8))
        y10 = self.relu10(self.conv10(y9))
        y11 = self.relu11(self.conv11(y10))
        y12 = self.relu12(self.conv12(y11))
        y13 = self.relu13(self.conv13(y12))

        y17 = self.relu17(self.conv17(y12))
        y18 = self.relu18(self.conv_transpose2d18(y17))
        c19 = torch.cat([y8, y18], 1)
        y20 = self.relu20(self.conv20(c19))
        y13 = y13.view(-1, 512*1*1)
        y20 = y20.view(-1, 256*2*2)

        return y13, y20


class TestNet12(nn.Module):
    """
    for 206 uavcut 1/8 32x32 t32 down input
    """
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 2, 3, stride=1, padding=1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(2, 2, 3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(2, 4, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(4, 4, 3, stride=2, padding=1, groups=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(4, 8, 3, stride=1, padding=1, groups=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(8, 8, 3, stride=2, padding=1, groups=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(8, 16, 3, stride=1, padding=1, groups=1)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=1)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(16, 32, 3, stride=1, padding=1, groups=1)
        self.relu8 = nn.ReLU()
        self.conv9 = nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=2)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(32, 64, 3, stride=1, padding=1, groups=2)
        self.relu10 = nn.ReLU()
        self.conv11 = nn.Conv2d(64, 128, 3, stride=1, padding=1, groups=4)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(128, 32, 1, stride=1, padding=0, groups=1)
        self.relu12 = nn.ReLU()
        self.conv13 = nn.Conv2d(32, 64, 3, stride=1, padding=1, groups=2)
        self.relu13 = nn.ReLU()

        self.conv17 = nn.Conv2d(32, 16, 1, stride=1, padding=0, groups=1)
        self.relu17 = nn.ReLU()
        self.conv_transpose2d18 = nn.ConvTranspose2d(16, 16, 3, padding=1, stride=2, output_padding=1, groups=1)
        self.relu18 = nn.ReLU()
        self.conv20 = nn.Conv2d(48, 32, 3, stride=1, padding=1, groups=2)
        self.relu20 = nn.ReLU()

    def forward(self, x):
        y0 = self.relu0(self.conv0(x))
        y1 = self.relu1(self.conv1(y0))
        y2 = self.relu2(self.conv2(y1))
        y3 = self.relu3(self.conv3(y2))
        y4 = self.relu4(self.conv4(y3))
        y5 = self.relu5(self.conv5(y4))
        y6 = self.relu6(self.conv6(y5))
        y7 = self.relu7(self.conv7(y6))
        y8 = self.relu8(self.conv8(y7))
        y9 = self.relu9(self.conv9(y8))
        y10 = self.relu10(self.conv10(y9))
        y11 = self.relu11(self.conv11(y10))
        y12 = self.relu12(self.conv12(y11))
        y13 = self.relu13(self.conv13(y12))

        y17 = self.relu17(self.conv17(y12))
        y18 = self.relu18(self.conv_transpose2d18(y17))
        c19 = torch.cat([y8, y18], 1)
        y20 = self.relu20(self.conv20(c19))
        y13 = y13.view(-1, 64*1*1)
        y20 = y20.view(-1, 32*2*2)

        return y13, y20