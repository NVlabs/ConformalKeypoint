import torch.nn as nn
import torch
import numpy as np
# from .layers import ConvBlock, ResBlock
from .layers import Residual

class Hourglass(nn.Module):
    def __init__(self, n, in_channels, out_channels):
        super(Hourglass, self).__init__()
        self.up1 = Residual(in_channels, 256)
        self.up2 = Residual(256, 256)
        self.up4 = Residual(256, out_channels)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.low1 = Residual(in_channels, 256)
        self.low2 = Residual(256, 256)
        self.low5 = Residual(256, 256)
        if n > 1:
            self.low6 = Hourglass(n-1, 256, out_channels)
        else:
            self.low6 = Residual(256, out_channels)
        self.low7 = Residual(out_channels, out_channels)
        # self.up5 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        up = self.up1(x)
        up = self.up2(up)
        up = self.up4(up)

        low = self.pool(x)
        low = self.low1(low)
        low = self.low2(low)
        low = self.low5(low)
        low = self.low6(low)
        low = self.low7(low)
        # low = self.up5(low)
        low = nn.functional.interpolate(low, scale_factor=2)

        return up + low

class Lin(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Lin, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.layer(x)

class StackedHourglass(nn.Module):
    def __init__(self, out_channels):
        super(StackedHourglass, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.r1 = Residual(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.r4 = Residual(128, 128)
        self.r5 = Residual(128, 128)
        self.r6 = Residual(128, 256)

        self.hg1 = Hourglass(4, 256, 512)

        self.l1 = Lin(512, 512)
        self.l2 = Lin(512, 256)

        self.out1 = nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0)

        self.out_return = nn.Conv2d(out_channels, 256+128, kernel_size=1, stride=1, padding=0)

        self.cat_conv = nn.Conv2d(256+128, 256+128, kernel_size=1, stride=1, padding=0)

        self.hg2 = Hourglass(4, 256+128, 512)

        self.l3 = Lin(512, 512)
        self.l4 = Lin(512, 512)

        self.out2 = nn.Conv2d(512, out_channels, 1, 1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.r1(x)
        pooled = self.pool(x)
        x = self.r4(pooled)
        x = self.r5(x)
        x = self.r6(x)

        # First hourglass
        x = self.hg1(x)

        # Linear layers to produce first set of predictions
        x = self.l1(x)
        x = self.l2(x)

        # First predicted heatmaps
        out1 = self.out1(x)
        out1_ = self.out_return(out1)

        joined = torch.cat([x, pooled], 1)
        joined = self.cat_conv(joined)
        int1 = joined + out1_

        hg2 = self.hg2(int1)

        l3 = self.l3(hg2)
        l4 = self.l4(l3)

        out2 = self.out2(l4)

        return out1, out2


    def num_trainable_parameters(self):
        trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in trainable_parameters])


