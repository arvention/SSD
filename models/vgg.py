import torch.nn as nn


base_config = {
    '300': [64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 'C',
            512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 'C',
            512, 512, 512, 'M',
            512, 512, 512],
}


def vgg(config, in_channels, batch_norm=False):
    layers = []

    for v in config:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2,
                                    stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2,
                                    stride=2,
                                    ceil_mode=True)]
        else:
            conv = nn.Conv2d(in_channels=in_channels,
                             out_channels=v,
                             kernel_size=3,
                             padding=1)
            if batch_norm:
                layers += [conv,
                           nn.BatchNorm2d(num_features=v),
                           nn.ReLU(inplace=True)]
            else:
                layers += [conv,
                           nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3,
                         stride=1,
                         padding=1)
    conv6 = nn.Conv2d(in_channels=512,
                      out_channels=1024,
                      kernel_size=3,
                      padding=6,
                      dilation=6)
    conv7 = nn.Conv2d(in_channels=1024,
                      out_channels=1024,
                      kernel_size=1)

    layers += [pool5,
               conv6,
               nn.ReLU(inplace=True),
               conv7,
               nn.ReLU(inplace=True)]

    return layers
