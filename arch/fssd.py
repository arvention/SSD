import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.detection import Detect
from arch.vgg import vgg, base_config


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
        self.bn = nn.BatchNorm2d(num_features=out_channels,
                                 eps=1e-5,
                                 momentum=0.01,
                                 affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size
        self.up_sample = nn.Upsample(size=(up_size, up_size),
                                     mode='bilinear') if up_size != 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x


class FSSD(nn.Module):

    """
    FSSD architecture
    """

    def __init__(self,
                 mode,
                 base,
                 extras,
                 fusion_module,
                 pyramid_module,
                 head,
                 anchors,
                 class_count):
        super(FSSD, self).__init__()
        self.mode = mode
        self.base = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.fusion_module = nn.ModuleList(fusion_module)
        self.bn = nn.BatchNorm2d(num_features=256 * len(self.fusion_module))
        self.pyramid_module = nn.ModuleList(pyramid_module)

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
        # apply vgg up to conv4_3 relu
        for i in range(23):
            x = self.base[i](x)
        sources.append(x)

        # apply vgg up to fc7
        for i in range(23, len(self.base)):
            x = self.base[i](x)
        sources.append(x)

        for layer in self.extras:
            x = F.relu(layer(x), inplace=True)
        sources.append(x)

        features = []
        assert len(self.fusion_module) == len(sources)
        for i, layer in enumerate(self.fusion_module):
            features.append(layer(sources[i]))

        features = torch.cat(features, 1)
        x = self.bn(features)

        feature_pyramid = []
        for layer in self.pyramid_module:
            x = layer(x)
            feature_pyramid.append(x)

        # apply multibox head to sources
        for (x, c, l) in zip(feature_pyramid, self.class_head, self.loc_head):
            class_preds.append(c(x).permute(0, 2, 3, 1).contiguous())
            loc_preds.append(l(x).permute(0, 2, 3, 1).contiguous())

        class_preds = torch.cat([pred.view(b, -1) for pred in class_preds], 1)
        loc_preds = torch.cat([pred.view(b, -1) for pred in loc_preds], 1)

        class_preds = class_preds.view(b, -1, self.class_count)
        loc_preds = loc_preds.view(b, -1, 4)

        if self.mode == "test":
            output = self.detect(
                self.softmax(class_preds),
                loc_preds,
                self.anchors
            )
        else:
            output = (
                class_preds,
                loc_preds
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


def get_fusion_module(config, vgg, extras):

    layers = []
    # conv4_3
    layers += [BasicConv(in_channels=vgg[24].out_channels,
                         out_channels=256,
                         kernel_size=1)]
    # fc_7
    layers += [BasicConv(in_channels=vgg[-2].out_channels,
                         out_channels=256,
                         kernel_size=1,
                         up_size=config)]

    layers += [BasicConv(in_channels=extras[-1].out_channels,
                         out_channels=256,
                         kernel_size=1,
                         up_size=config)]

    return layers


def get_pyramid_module(config):

    layers = []

    for layer in config:
        layers += [BasicConv(in_channels=layer[0],
                             out_channels=layer[1],
                             kernel_size=layer[2],
                             stride=layer[3],
                             padding=layer[4])]

    return layers


def multibox(config, class_count):
    class_layers = []
    loc_layers = []

    for in_channels, num_anchors in enumerate(config):
        class_layers += [nn.Conv2d(in_channels=in_channels,
                                   out_channels=num_anchors * class_count,
                                   kernel_size=3,
                                   padding=1)]
        loc_layers += [nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_anchors * 4,
                                 kernel_size=3,
                                 padding=1)]

    return class_layers, loc_layers


extras_config = {
    '300': [256, 512, 128, 'S', 256],
    '512': [256, 512, 128, 'S', 256]
}
fusion_config = {
    '300': 38,
    '512': 64
}
pyramid_config = {
    '300': [[256 * 3, 512, 3, 1, 1],
            [512, 512, 3, 2, 1],
            [512, 256, 3, 2, 1],
            [256, 256, 3, 2, 1],
            [256, 256, 3, 1, 0],
            [256, 256, 3, 1, 0]],

    '512': [[256 * 3, 512, 3, 1, 1],
            [512, 512, 3, 2, 1],
            [512, 256, 3, 2, 1],
            [256, 256, 3, 2, 1],
            [256, 256, 3, 2, 1],
            [256, 256, 3, 2, 1],
            [256, 256, 4, 1, 1]]
}
mbox_config = {
    '300': [(512, 6),
            (512, 6),
            (256, 6),
            (256, 6),
            (256, 4),
            (256, 4)],
    '512': [(512, 6),
            (512, 6),
            (256, 6),
            (256, 6),
            (256, 6),
            (256, 4),
            (256, 4)]
}


def build_fssd(mode, new_size, anchors, class_count):

    base = vgg(config=base_config[str(new_size)],
               in_channels=3)

    extras = get_extras(config=extras_config[str(new_size)],
                        in_channels=1024)

    fusion_module = get_fusion_module(config=fusion_config[str(new_size)],
                                      vgg=base,
                                      extras=extras)

    pyramid_module = get_pyramid_module(config=pyramid_config[str(new_size)])

    head = multibox(config=mbox_config[str(new_size)],
                    class_count=class_count)

    return FSSD(mode, base, extras, fusion_module,
                pyramid_module, head, anchors, class_count)
