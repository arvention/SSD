import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.detection import Detect
from models.vgg import vgg, base_config
from layers.block import BasicConv
from utils.init import xavier_init


class ShuffleSSD(nn.Module):

    """
    Shuffle SSD architecture
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
        super(ShuffleSSD, self).__init__()
        self.mode = mode
        self.base = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.fusion_module = nn.ModuleList(fusion_module)
        self.bn = nn.BatchNorm2d(num_features=(512 * len(self.fusion_module)))
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
        for i, layer in enumerate(self.get_fusion_module):
            features.append(layer(sources[i]))

        features = torch.cat(features, 1)  # 10x10x(512*3)
        x = self.bn(features)

        feature_pyramid = []
        y = x
        for i, _ in enumerate(self.pyramid_module):
            if i < 3:
                y = self.pyramid_module[i](y)
                feature_pyramid.append(y)
            else:
                x = self.pyramid_module[i](x)
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

    def init_weights(self, model_save_path, basenet):
        if basenet:
            weights_path = osp.join(model_save_path, basenet)
            vgg_weights = torch.load(weights_path)
            self.base.load_state_dict(vgg_weights)
        else:
            self.base.apply(fn=xavier_init)
        self.extras.apply(fn=xavier_init)
        self.fusion_module.apply(fn=xavier_init)
        self.pyramid_module.apply(fn=xavier_init)
        self.class_head.apply(fn=xavier_init)
        self.loc_head.apply(fn=xavier_init)

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


def get_fusion_module(vgg, extras):

    layers = []
    # conv4_3
    layers += [nn.Sequential(BasicConv(in_channels=vgg[24].out_channels,
                                       out_channels=256,
                                       kernel_size=2,
                                       stride=2),
                             BasicConv(in_channels=256,
                                       out_channels=512,
                                       kernel_size=2,
                                       stride=2,
                                       padding=1))]
    # fc_7
    layers += [BasicConv(in_channels=vgg[-2].out_channels,
                         out_channels=512,
                         kernel_size=2,
                         stride=2,
                         padding=1)]

    layers += [BasicConv(in_channels=extras[-1].out_channels,
                         out_channels=512,
                         kernel_size=1)]

    return layers


def get_pyramid_module(config):

    layers = []

    layers += [nn.PixelShuffle(upscale_factor=2)]

    layers += [nn.Sequential(BasicConv(in_channels=384,
                                       out_channels=config,
                                       kernel_size=2,
                                       stride=2),
                             nn.PixelShuffle(upscale_factor=4))]

    layers += [nn.Sequential(BasicConv(in_channels=192,
                                       out_channels=config,
                                       kernel_size=2,
                                       stride=2),
                             nn.PixelShuffle(upscale_factor=3))]

    layers += [BasicConv(in_channels=config,
                         out_channels=256,
                         kernel_size=1,
                         stride=1)]
    layers += [BasicConv(in_channels=256,
                         out_channels=256,
                         kernel_size=2,
                         stride=2)]
    layers += [BasicConv(in_channels=256,
                         out_channels=256,
                         kernel_size=2,
                         stride=2,
                         padding=1)]
    layers += [BasicConv(in_channels=256,
                         out_channels=256,
                         kernel_size=2,
                         stride=2)]

    return layers


def multibox(config, class_count):
    class_layers = []
    loc_layers = []

    for in_channels, num_anchors in config:
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
    '300': [256, 512, 128, 'S', 256]
}
pyramid_config = {
    '300': (512 * 3)
}
mbox_config = {
    '300': [(384, 6),
            (192, 6),
            (128, 6),
            (256, 6),
            (256, 6),
            (256, 6),
            (256, 6)]
}


def build_shuffle_ssd(mode, new_size, anchors, class_count):

    base = vgg(config=base_config[str(new_size)],
               in_channels=3)

    extras = get_extras(config=extras_config[str(new_size)],
                        in_channels=1024)

    fusion_module = get_fusion_module(vgg=base,
                                      extras=extras)

    pyramid_module = get_pyramid_module(config=pyramid_config[str(new_size)])

    head = multibox(config=mbox_config[str(new_size)],
                    class_count=class_count)

    return ShuffleSSD(mode, base, extras, fusion_module, pyramid_module,
                      head, anchors, class_count)
