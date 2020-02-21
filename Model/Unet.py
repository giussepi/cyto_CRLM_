from __future__ import absolute_import


import torch
from torch import nn
import torch.nn.functional as F

"""
reference
https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py
"""



class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm,kch_size=512):
        super(UNetUpBlock, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=3,
                                         stride=2)
        
        if in_size== kch_size and out_size==kch_size:
            self.conv_block = UNetConvBlock(in_size*2, out_size, padding, batch_norm)
        else:
            self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)
        
    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(up, bridge.shape[2:])
        out = torch.cat([bridge, crop1], 1)
        out = self.conv_block(out)

        return out

class Unet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, channels = [64,128,256,512,512,512,512], padding=True,
                 batch_norm=True, ):
        super(Unet, self).__init__()

        self.padding = padding
        self.depth = len(channels)
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        
        for i in range(self.depth):
            self.down_path.append(UNetConvBlock(prev_channels, channels[i],
                                                padding, batch_norm))
            prev_channels = channels[i]

        self.up_path = nn.ModuleList()
        for i in reversed(range(self.depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, channels[i], 
                                            padding, batch_norm))
            prev_channels = channels[i]

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)
        #self.softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Sigmoid()   # To use the Crossentropy2d don't need the 
        
    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)
        up_blocks = [] 
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])
            up_blocks.append(x)
        return self.softmax(self.last(x))
