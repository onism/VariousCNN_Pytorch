'''
https://arxiv.org/pdf/1707.03718.pdf

   Input
     |
     |
     max-pool
     |
     |                                full_conv  
     conv()                             conv
     |                                full_conv  
     |                                  |
     Encoder 1                        Decoder 1  
     |__________________________________+
     |                                  |
     Encoder 2                         Decoder 2 
     |__________________________________ +
     |                                   |
     EnCoder 3                        Decoder 3
     |___________________________________+
     |                                   |
     Encoder 4                         Decoder 4 
      |__________________________________|

 LinkNet uses ResNet18 as its encode
'''

import torch 
import torch.nn as nn 
from torch.autograd import Variable 
from torchvision.models import resnet
from torchsummary import summary


class Decoder(nn.Module):
    '''
    Fig 3
    Input ---> conv( kernel=1, input=m, output=m/4) --> full_conv(kernel=3, input=m/4,output=m/4) --> conv(kernel=1, input=m/4, output=n)
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0,output_padding=0,  bias=False):
        super(Decoder, self).__init__()
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=in_channels//4, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=in_channels//4, out_channels=in_channels//4, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=output_padding, bias=bias),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels//4, out_channels=out_channels, kernel_size=1, stride=1,padding=0,bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.decoder_conv(x)

class LinkNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(LinkNet, self).__init__()
        base = resnet.resnet18(pretrained=True)
        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2 
        self.encoder3 = base.layer3 
        self.encoder4 = base.layer4 

        self.decoder1 = Decoder(in_channels=64, out_channels=64, kernel_size=3,stride=1, padding=1,output_padding=0)
        self.decoder2 = Decoder(in_channels=128, out_channels=64, kernel_size=3,stride=2, padding=1,output_padding=1)
        self.decoder3 = Decoder(in_channels=256, out_channels=128, kernel_size=3,stride=2, padding=1,output_padding=1)
        self.decoder4 = Decoder(in_channels=512, out_channels=256, kernel_size=3,stride=2, padding=1,output_padding=1)

        self.final_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=out_channels, kernel_size=2, stride=2,padding=0)
        )
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.in_block(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)

        y = self.final_block(d1)
        y = self.lsm(y)
        return y     


model = LinkNet(3, 1)
summary(model,(3,512,1024))