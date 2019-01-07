import os
import os.path as osp
import torch
import torch.nn as nn
from layers.detection2 import Detect
from models.vgg import vgg, base_config
from layers.block import BasicConv, BasicRFB, BasicRFB_a
from utils.init import xavier_init


class RFBNet(nn.Module):

    """
    RFBNet architecture
    """

    def __init__(self,
                 mode,
                 new_size,
                 base,
                 extras,
                 head,
                 anchors,
                 class_count):
        super(RFBNet, self).__init__()
        self.mode = mode
        self.base = nn.ModuleList(base)
        self.norm = BasicRFB_a(512, 512, stride=1, scale=1.0)
        self.extras = nn.ModuleList(extras)

        if new_size == 300:
            self.indicator = 3
        elif new_size == 512:
            self.indicator = 5

        self.class_head = nn.ModuleList(head[0])
        self.loc_head = nn.ModuleList(head[1])
        self.anchors = anchors
        self.class_count = class_count

        if mode == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(class_count)

    def forward(self, x):
        sources = []
        class_preds = []
        loc_preds = []

        b, _, _, _ = x.shape
        # apply vgg up to conv4_3 relu
        for i in range(23):
            x = self.base[i](x)

        s = self.norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for i in range(23, len(self.base)):
            x = self.base[i](x)

        for i, layer in enumerate(self.extras):
            x = layer(x)
            if i < self.indicator or i % 2 == 0:
                sources.append(x)

        # apply multibox head to sources
        for (x, c, l) in zip(sources, self.class_head, self.loc_head):
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
        self.norm.apply(fn=xavier_init)
        self.extras.apply(fn=xavier_init)
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


def get_extras(config, in_channels, size, batch_norm=False):
    layers = []

    for k, v in enumerate(config):
        if in_channels != 'S':
            if v == 'S':
                if in_channels == 256 and size == 512:
                    layers += [BasicRFB(in_channels=in_channels,
                                        out_channels=config[k + 1],
                                        stride=2,
                                        scale=1.0,
                                        visual=1)]
                else:
                    layers += [BasicRFB(in_channels=in_channels,
                                        out_channels=config[k + 1],
                                        stride=2,
                                        scale=1.0,
                                        visual=2)]
            else:
                layers += [BasicRFB(in_channels=in_channels,
                                    out_channels=v,
                                    scale=1.0,
                                    visual=2)]
        in_channels = v

    if size == 512:
        layers += [BasicConv(in_channels=256,
                             out_channels=128,
                             kernel_size=1,
                             stride=1)]
        layers += [BasicConv(in_channels=128,
                             out_channels=256,
                             kernel_size=4,
                             stride=1,
                             padding=1)]

    elif size == 300:
        layers += [BasicConv(in_channels=256,
                             out_channels=128,
                             kernel_size=1,
                             stride=1)]
        layers += [BasicConv(in_channels=128,
                             out_channels=256,
                             kernel_size=3,
                             stride=1)]
        layers += [BasicConv(in_channels=256,
                             out_channels=128,
                             kernel_size=1,
                             stride=1)]
        layers += [BasicConv(in_channels=128,
                             out_channels=256,
                             kernel_size=3,
                             stride=1)]

    return layers


def multibox(config, base, extra_layers, size, class_count):
    class_layers = []
    loc_layers = []
    vgg_source = [-2]

    for k, v in enumerate(vgg_source):
        if k == 0:
            class_layers += [nn.Conv2d(in_channels=512,
                                       out_channels=(config[k] * class_count),
                                       kernel_size=3,
                                       padding=1)]
            loc_layers += [nn.Conv2d(in_channels=512,
                                     out_channels=(config[k] * 4),
                                     kernel_size=3,
                                     padding=1)]
        else:
            class_layers += [nn.Conv2d(in_channels=base[v].out_channels,
                                       out_channels=(config[k] * class_count),
                                       kernel_size=3,
                                       padding=1)]
            loc_layers += [nn.Conv2d(in_channels=base[v].out_channels,
                                     out_channels=(config[k] * 4),
                                     kernel_size=3,
                                     padding=1)]

    i = 1
    indicator = 0
    if size == 300:
        indicator = 3
    elif size == 512:
        indicator = 5

    for k, v in enumerate(extra_layers):
        if k < indicator or k % 2 == 0:
            class_layers += [nn.Conv2d(in_channels=v.out_channels,
                                       out_channels=(config[i] * class_count),
                                       kernel_size=3,
                                       padding=1)]
            loc_layers += [nn.Conv2d(in_channels=v.out_channels,
                                     out_channels=(config[i] * 4),
                                     kernel_size=3,
                                     padding=1)]
            i += 1

    return class_layers, loc_layers


extras_config = {
    '300': [1024, 'S', 512, 'S', 256],
    '512': [1024, 'S', 512, 'S', 256, 'S', 256, 'S', 256],
}
mbox_config = {
    '300': [4, 6, 6, 6, 4, 4],
    '512': [4, 6, 6, 6, 6, 4, 4]
}


def build_rfbnet(mode, new_size, anchors, class_count):

    base = vgg(config=base_config[str(new_size)],
               in_channels=3)

    extras = get_extras(config=extras_config[str(new_size)],
                        in_channels=1024,
                        size=new_size)

    head = multibox(config=mbox_config[str(new_size)],
                    base=base,
                    extra_layers=extras,
                    size=new_size,
                    class_count=class_count)

    return RFBNet(mode, new_size, base, extras, head, anchors, class_count)
