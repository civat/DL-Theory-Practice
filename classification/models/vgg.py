import torch
import torch.nn as nn

import register
from classification import utils
from nn_module.conv.convs import Conv2d
from classification.models.plainnet import PlainNet


@register.name_to_model.register("VGG")
class VGG(nn.Module):

    def __init__(self, in_channels, hidden_channels, n_blocks_list, stride_list, stride_factor, pool_size,
                 fc_in_dim, fc_out_dim, num_classes, last_act, conv=Conv2d):
        super().__init__()
        assert len(n_blocks_list) > 0
        assert len(n_blocks_list) == len(stride_list)
        self.backbone, out_channels = VGG.make_backbone(n_blocks_list, stride_list, in_channels, hidden_channels,
                                                        stride_factor, pool_size, conv)
        self.last_act = last_act()

        # For binary classification task, we use BCE loss so only one output logit is needed.
        self.num_classes = num_classes
        if num_classes == 2:
            num_classes = 1
        if num_classes > 0:
            self.fc = nn.Sequential(
                nn.Linear(fc_in_dim, fc_out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(fc_out_dim, fc_out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(fc_out_dim, num_classes)
            )

    def forward(self, x):
        output = self.last_act(self.backbone(x))
        if self.num_classes > 0:
            output = torch.flatten(output, start_dim=1)
            output = self.fc(output)
        return output

    @staticmethod
    def make_plain_part(in_channels, hidden_channels, stride, n_blocks, pool_size, conv):
        layers = PlainNet.make_plain_part(in_channels, hidden_channels, 1, n_blocks, conv)
        layers.append(nn.MaxPool2d(pool_size, stride=stride, padding=int(pool_size // 2)))
        return layers

    @staticmethod
    def make_backbone(n_blocks_list, stride_list, in_channels, hidden_channels, stride_factor, pool_size, conv):
        convs = []
        stride_factors = [PlainNet._get_stride_factor(stride, stride_factor) for stride in stride_list]

        for i, (n_blocks, stride, stride_factor) in enumerate(zip(n_blocks_list, stride_list, stride_factors)):
            if i != 0:
                hidden_channels = int(in_channels * stride_factor)
            convs += VGG.make_plain_part(in_channels, hidden_channels, stride, n_blocks, pool_size, conv)
            in_channels = hidden_channels

        convs = nn.Sequential(*convs)
        return convs, hidden_channels

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
            "pool_size"      : 3,
            "fc_in_dim"      : 1024,
            "fc_out_dim"     : 4096,
            "num_classes"    : 10,
            "last_act"       : act,
            "conv"           : conv,
        }
        default_params = utils.set_params(default_params, configs, excluded_keys=["conv", "last_act"])
        return VGG(**default_params)