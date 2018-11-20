import torch
import torch.nn as nn


class BasicConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=False,
                 bias=True,
                 up_size=0):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels,
                                     eps=1e-5,
                                     momentum=0.01,
                                     affine=True)
        self.relu = None
        if relu:
            self.relu = nn.ReLU(inplace=True)

        self.up_sample = None
        if up_size != 0:
            self.up_sample = nn.Upsample(size=(up_size, up_size),
                                         mode='bilinear')

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x

