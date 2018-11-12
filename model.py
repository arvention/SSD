import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.l2_norm import L2Norm
from layers.detection import Detect


class SSD(nn.Module):

    """
    SSD architecture
    """

    def __init__(self,
                 mode,
                 base,
                 extras,
                 head,
                 anchors,
                 class_count):
        super(SSD, self).__init__()

        self.mode = mode
        self.base = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.class_head = nn.ModuleList(head[0])
        self.loc_head = nn.ModuleList(head[1])
        self.anchors = anchors
        self.class_count = class_count

        if mode == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(class_count, 200, 0.01, 0.45)

    def forward(self, x):
        sources = []
        class_preds = []
        loc_preds = []

        b, _, _, _ = x.shape
        y = x
        # apply vgg up to conv4_3 relu
        for i in range(23):
            y = self.base[i](y)

        s = self.L2Norm(y)
        sources.append(s)

        # apply vgg up to fc7
        for i in range(23, len(self.base)):
            y = self.base[i](y)
        sources.append(y)

        # apply extras
        for i, layer in enumerate(self.extras):
            y = F.relu(layer(y), inplace=True)
            if i % 2 == 1:
                sources.append(y)

        # apply multibox head to sources
        for (y, c, l) in zip(sources, self.class_head, self.loc_head):
            class_preds.append(c(y).permute(0, 2, 3, 1).contiguous())
            loc_preds.append(l(y).permute(0, 2, 3, 1).contiguous())

        class_preds = torch.cat([pred.view(b, -1) for pred in class_preds], 1)
        loc_preds = torch.cat([pred.view(b, -1) for pred in loc_preds], 1)

        class_preds = class_preds.view(b, -1, self.class_count)
        loc_preds = loc_preds.view(b, -1, 4)

        if self.mode == "test":
            output = self.detect(
                self.softmax(class_preds),
                loc_preds,
                self.anchors.type(type(x.data))
            )
        else:
            output = (
                class_preds,
                loc_preds,
                self.anchors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


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


def get_extras(config, in_channels, batch_norm=False):
    layers = []
    flag = False

    for k, v in enumerate(config):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels=in_channels,
                                     out_channels=config[k + 1],
                                     kernel_size=(1, 3)[flag],
                                     stride=2,
                                     padding=1)]
            else:
                layers += [nn.Conv2d(in_channels=in_channels,
                                     out_channels=v,
                                     kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v

    return layers


def multibox(vgg, extra_layers, config, class_count):
    class_layers = []
    loc_layers = []
    vgg_source = [21, -2]

    for k, v in enumerate(vgg_source):
        class_layers += [nn.Conv2d(in_channels=vgg[v].out_channels,
                                   out_channels=config[k] * class_count,
                                   kernel_size=3,
                                   padding=1)]
        loc_layers += [nn.Conv2d(in_channels=vgg[v].out_channels,
                                 out_channels=config[k] * 4,
                                 kernel_size=3,
                                 padding=1)]

    for k, v in enumerate(extra_layers[1::2], start=2):
        class_layers += [nn.Conv2d(in_channels=v.out_channels,
                                   out_channels=config[k] * class_count,
                                   kernel_size=3,
                                   padding=1)]
        loc_layers += [nn.Conv2d(in_channels=v.out_channels,
                                 out_channels=config[k] * 4,
                                 kernel_size=3,
                                 padding=1)]

    return class_layers, loc_layers


base_config = {
    '300': [64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 'C',
            512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras_config = {
    '300': [256, 'S',
            512, 128, 'S',
            256, 128, 256, 128, 256],
    '512': [],
}
mbox_config = {
    '300': [4, 6, 6, 6, 4, 4],
    '512': [],
}


def build_ssd(mode, new_size, anchors, class_count):

    base = vgg(config=base_config[str(new_size)],
               in_channels=3)
    extras = get_extras(config=extras_config[str(new_size)],
                        in_channels=1024)

    head = multibox(vgg=base,
                    extra_layers=extras,
                    config=mbox_config[str(new_size)],
                    class_count=class_count)

    return SSD(mode, base, extras, head, anchors, class_count)
