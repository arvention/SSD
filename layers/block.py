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


class BasicRFB(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 scale=0.1,
                 visual=1):
        super(BasicRFB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.scale = scale
        self.visual = visual

        self.inter_channnels = in_channels // 8

        self.branch0 = self.get_branch0()
        self.branch1 = self.get_branch1()
        self.branch2 = self.get_branch2()

        self.conv_linear = BasicConv(in_channels=(6 * self.inter_channnels),
                                     out_channels=self.out_channels,
                                     kernel_size=1,
                                     stride=1,
                                     relu=False)

        self.shortcut = BasicConv(in_channels=self.in_channels,
                                  out_channels=self.out_channels,
                                  kernel_size=1,
                                  stride=self.stride,
                                  relu=False)

        self.relu = nn.ReLU(inplace=True)

    def get_branch0(self):

        layers = []

        layers += [BasicConv(in_channels=self.in_channels,
                             out_channels=(2 * self.inter_channnels),
                             kernel_size=1,
                             stride=self.stride)]

        layers += [BasicConv(in_channels=(2 * self.inter_channnels),
                             out_channels=(2 * self.inter_channnels),
                             kernel_size=3,
                             stride=1,
                             padding=self.visual,
                             dilation=self.visual,
                             relu=False)]

        return nn.Sequential(*layers)

    def get_branch1(self):

        layers = []

        layers += [BasicConv(in_channels=self.in_channels,
                             out_channels=self.inter_channnels,
                             kernel_size=1,
                             stride=1)]

        layers += [BasicConv(in_channels=self.inter_channnels,
                             out_channels=(2 * self.inter_channnels),
                             kernel_size=(3, 3),
                             stride=self.stride,
                             padding=(1, 1))]

        layers += [BasicConv(in_channels=(2 * self.inter_channnels),
                             out_channels=(2 * self.inter_channnels),
                             kernel_size=3,
                             stride=1,
                             padding=(self.visual + 1),
                             dilation=(self.visual + 1),
                             relu=False)]

        return nn.Sequential(*layers)

    def get_branch2(self):

        layers = []

        layers += [BasicConv(in_channels=self.in_channels,
                             out_channels=self.inter_channnels,
                             kernel_size=1,
                             stride=1)]

        layers += [BasicConv(in_channels=self.inter_channnels,
                             out_channels=(self.inter_channnels // 2 * 3),
                             kernel_size=3,
                             stride=1,
                             padding=1)]

        layers += [BasicConv(in_channels=(self.inter_channnels // 2 * 3),
                             out_channels=(2 * self.inter_channnels),
                             kernel_size=3,
                             stride=self.stride,
                             padding=1)]

        layers += [BasicConv(in_channels=(2 * self.inter_channnels),
                             out_channels=(2 * self.inter_channnels),
                             kernel_size=3,
                             stride=1,
                             padding=(2 * self.visual + 1),
                             dilation=(2 * self.visual + 1),
                             relu=False)]

        return nn.Sequential(*layers)

    def forward(self, x):
        y0 = self.branch0(x)
        y1 = self.branch1(x)
        y2 = self.branch2(x)

        y = torch.cat((y0, y1, y2), 1)
        y = self.conv_linear(y)
        short = self.shortcut(x)
        y = y * self.scale + short
        y = self.relu(y)

        return y
