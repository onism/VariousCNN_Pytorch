import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import math 

# copied from torchvision/models/resnet.py

def initvars(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2./n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class scale_and_shift(nn.Module):
    def __init__(self):
        super(scale_and_shift, self).__init__()

        self.alpha = torch.ones(1) 
        self.beta = torch.zeros(1) 

    def forward(self, input):
        return input * self.alpha + self.beta


class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()
        self.scale_and_shift = scale_and_shift()
         
    
    def forward(self, x):
        # Negation simply multiplies −1 to the output of Convolution.
        x = torch.cat( (x, -x), 1 )
        # Scale / Shift applies trainable weight and bias to each chan ? should be random?
        x = self.scale_and_shift(x)
        x = F.relu(x)
        return x

class CReLUConv(nn.Module):
    def __init__(self, in_channels, out_channels=1, kernel_size=1, stride=1, padding = 0):
        super(CReLUConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.crelu = CReLU()
    def forward(self, input):
        x = self.bn(self.conv(x))
        x = self.crelu(x)
        return x 

class bn_scale_relu(nn.Module):
    def __init__(self, in_channels):
        super(bn_scale_relu, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.scale_and_shift = scale_and_shift()
        self.relu = nn.ReLU()
    
    def forward(self, input):
        x = self.bn(input)
        x = self.scale_and_shift(x)
        x = self.relu(x)
        return x 
    
class res_crelu(nn.Module):
    def __init__(self, in_channels, middel_channels, out_channels, kernel_size, stride, padding, bsr, proj):
        super(res_crelu, self).__init__()
        self.bsr = bsr 
        self.proj = proj 
        self.bn_scale_relu_input = bn_scale_relu(in_channels)
        self.bn_scale_relu_conv1 = bn_scale_relu(middel_channels[0])
        if self.proj:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, 1, stride)
        self.conv1 = nn.Conv2d(in_channels, middel_channels[0], 1, stride, padding=0)
        self.conv2 = nn.Conv2d(middel_channels[0], middel_channels[1], kernel_size, 1, padding)
        self.bn = nn.BatchNorm2d(middel_channels[1])
        self.crelu = CReLU()
        self.conv3 = nn.Conv2d(2*middel_channels[1], out_channels, 1, 1, 0)
    
    def forward(self, input):
        if self.bsr:
            x = self.bn_scale_relu_input(input)
        else:
            x = input 
        
        if self.proj:
            shortcut = self.shortcut_conv(input)
        else:
            shortcut = input 
        x = self.conv1(x)
        x = self.bn_scale_relu_conv1(x)
        x = self.bn(self.conv2(x))
        x = self.crelu(x)
        x = self.conv3(x)
        out = x + shortcut
        return out 

class Inception(nn.Module):
    def __init__(self, in_channels, middel_channels, out_channels, kernel, stride, proj, last=False):
        super(Inception, self).__init__()
        self.stride = stride 
        self.last = last 
        self.proj = proj 
        if proj:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, 1, stride )
        self.bsr = bn_scale_relu(in_channels)
        # conv kernel 1 * 1
        self.conv_a = nn.Conv2d(in_channels, middel_channels[0], 1, 1, 0)
        # conv kernel 1 * 1 --> 3 * 3
        self.conv_b1 = nn.Conv2d(in_channels, middel_channels[1][0],1, 1, 0)
        self.conv_b2 = nn.Conv2d(middel_channels[1][0], middel_channels[1][1], 3, 1, 1)
        # conv kernel 1 * 1 --> 3 * 3 --> 3 * 3
        self.conv_c1 = nn.Conv2d(in_channels, middel_channels[2][0], 1, stride, 0)
        self.conv_c2 = nn.Conv2d(middel_channels[2][0], middel_channels[2][1], 3, 1, 1)
        self.conv_c3 = nn.Conv2d(middel_channels[2][1], middel_channels[2][2],3, 1, 1)

# too complexity, need more time 