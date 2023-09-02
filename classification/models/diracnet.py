from torch import nn

import register
from classification import utils
from classification.models.vgg import VGG
from classification.models.plainnet import PlainNet
from nn_module.conv.convs import Conv2d


@register.name_to_model.register("DiracNet")
class DiracNet(nn.Module):

    def __init__(self, in_channels, hidden_channels, n_blocks_list, stride_list, stride_factor, num_classes,
                 last_act, pool_size, k, conv=Conv2d):
        super().__init__()
        assert len(n_blocks_list) > 0
        assert len(n_blocks_list) == len(stride_list)
        self.backbone = [conv(in_channels, hidden_channels, stride=1)]

        # Define the first to last (exclude) groups
        vgg_backbone, hidden_channels = VGG.make_backbone(n_blocks_list[:-1], stride_list[:-1], hidden_channels,
                                                          hidden_channels * k, stride_factor, pool_size, conv)
        self.backbone += vgg_backbone

        sf = PlainNet._get_stride_factor(stride_list[-1], stride_factor)
        out_channels = int(hidden_channels * sf)
        last_group = PlainNet.make_plain_part(hidden_channels, out_channels, 1, n_blocks_list[-1], conv)
        self.backbone += last_group
        self.backbone = nn.Sequential(*self.backbone)
        self.last_act = last_act()

        self.num_classes = num_classes
        if num_classes == 2:
            num_classes = 1
        if num_classes > 0:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        output = self.last_act(self.backbone(x))
        if self.num_classes > 0:
            output = self.avg_pool(output)
            output = output.view(output.size(0), -1)
            output = self.fc(output)
        return output

    @staticmethod
    def make_network(configs):
        conv = register.get_conv(configs, "conv")
        act = register.get_activation(configs["last_act"])

        default_params = {
            "in_channels"    : 3,
            "hidden_channels": 64,
            "n_blocks_list"  : [2, 2, 2, 2],
            "stride_list"    : [2, 1, 1, 1],
            "stride_factor"  : 2,
            "num_classes"    : 10,
            "last_act"       : act,
            "pool_size"      : 3,
            "k"              : 5,
            "conv"           : conv,
        }
        default_params = utils.set_params(default_params, configs, excluded_keys=["conv", "last_act"])
        return DiracNet(**default_params)