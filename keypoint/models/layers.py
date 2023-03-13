import torch.nn as nn

# Wrapper around Conv2d
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block =  nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels//2, kernel_size=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(True),
            nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(True),
            nn.Conv2d(out_channels//2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.block(x)

class SkipLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipLayer, self).__init__()
        if in_channels != out_channels:
            self.layer =  nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.layer = None

    def forward(self, x):
        if self.layer is None:
            return x
        else:
            return self.layer(x)

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.skip = SkipLayer(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x) + self.skip(x)
