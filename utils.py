import torch
import torch.nn as nn
import torch.nn.functional as F

class UpSample(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(UpSample, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = nn.ReLU(True)(x)
        return x

class DownSample(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = nn.ReLU(True)(x)
        return x

class SameRes(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(SameRes, self).__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = nn.ReLU(True)(x)
        return x
