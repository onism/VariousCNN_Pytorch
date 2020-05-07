'''
Original Unet

Input ---conv --conv -- conv---------------------------------------------copy---------------------------------------------------conv---conv--conv--output
                         |                                                                                                        |
                         | maxpool                                                                                               up
                         |                                                                                                        |
                         conv -- conv -- conv-------------------------------copy-----------------------------------conv---conv---conv
                                           |                                                                          |  
                                           |maxpool                                                                  up
                                           |                                                                          |
                                           conv -- conv -- conv----------------copy----------------------conv--conv--conv
                                                             |                                           |    
                                                             |maxpool                                   up
                                                             |                                           |
                                                             conv--conv--conv ----copy -----conv--conv--conv
                                                                           |                  |
                                                                           |maxpool           |up conv  
                                                                           |                  |
                                                                           conv ---conv ---conv



'''


import torch 
from torch import nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torchsummary import summary



class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)



class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.up(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        filters = [64, 64*2, 64*4, 64*8,64*16]
        self.in_conv = conv_block(in_channels, filters[0])
        self.down_block1 = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(filters[0], filters[1])
        )

        self.down_block2 = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(filters[1], filters[2])
        )

        self.down_block3 = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(filters[2], filters[3])
        )

        self.down_block4 = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(filters[3], filters[4])
        )

        self.up5 = up_conv(filters[4], filters[3])
        self.up_conv5 = conv_block(filters[4], filters[3])

        self.up4 = up_conv(filters[3], filters[2])
        self.up_conv4 = conv_block(filters[3], filters[2])

        self.up3 = up_conv(filters[2], filters[1])
        self.up_conv3 = conv_block(filters[2], filters[1])

        self.up2 = up_conv(filters[1], filters[0])
        self.up_conv2 = conv_block(filters[1], filters[0])

        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1,stride=1,padding=0)
    
    def forward(self, x):
        e1 = self.in_conv(x)
        e2 = self.down_block1(e1)
        e3 = self.down_block2(e2)
        e4 = self.down_block3(e3)
        e5 = self.down_block4(e4)

        # up and concat
        d5 = self.up5(e5)
        d5 = torch.cat((e4,d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((e2,d3),dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e1, d2),dim=1)
        d2 = self.up_conv2(d2)

        out = self.final_conv(d2)
        # need activation?  
        return out     


model = UNet(3, 1)
summary(model,(3,512,1024))