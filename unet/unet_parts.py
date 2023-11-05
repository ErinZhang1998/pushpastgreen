""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)


class DownMaxpoolSingleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size),
            SingleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class DownRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then single conv"""

    def __init__(self, in_channels, out_channels, conv_in_channels, conv_out_channels, kernel_size, bilinear=True):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2)
        self.conv = SingleConv(conv_in_channels, conv_out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UpNoSkipConnection(nn.Module):
    """Upscaling then single conv"""

    def __init__(self, in_channels, out_channels, conv_out_channels, kernel_size, scale_factor=2,bilinear=True,padding=0):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor)
        self.conv = SingleConv(out_channels, conv_out_channels)

    def forward(self, x1, height=None, width=None):
        x1 = self.up(x1)
        # input is CHW
        if height and width:
            diffY = height - x1.size()[2]
            diffX = width - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(x1)


class ConvTransposeUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, conv_out_channels, scale_factor):
        super(ConvTransposeUpsample, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=scale_factor, padding=1, output_padding=scale_factor-1)
        self.conv = SingleConv(out_channels, conv_out_channels)

    def forward(self, x):
        x1 = self.conv_transpose(x)
        return self.conv(x1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
