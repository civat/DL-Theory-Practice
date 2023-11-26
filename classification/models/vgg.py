import torch
import torch.nn as nn
from collections import OrderedDict

import register
from classification import utils
from nn_module.conv.convs import Conv2d
from classification.models.plainnet import PlainNet


@register.name_to_model.register("VGG")
class VGG(nn.Module):

    def __init__(self, in_channels, hidden_channels, n_blocks_list, stride_list, stride_factor, pool_size,
                 fc_in_dim, fc_out_dim, num_classes, last_act, conv=Conv2d, out_feats=None):
        super().__init__()
        assert len(n_blocks_list) > 0
        assert len(n_blocks_list) == len(stride_list)
        assert isinstance(stride_factor, int) or len(stride_factor) == len(n_blocks_list)

        if isinstance(stride_factor, int):
            stride_factor = [stride_factor] * len(n_blocks_list)
            stride_factor = [PlainNet._get_stride_factor(stride, sf) for stride, sf in zip(stride_list, stride_factor)]

        self.backbone, self.out_channels = VGG.make_backbone(n_blocks_list, stride_list, in_channels, hidden_channels,
                                                             stride_factor, pool_size, conv)
        self.last_act = last_act()
        self.out_feats = out_feats

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
        if self.out_feats is None or self.out_feats == "None":
            output = self.last_act(self.backbone(x))
            if self.num_classes > 0:
                output = torch.flatten(output, start_dim=1)
                output = self.fc(output)
            return output
        else:
            outputs = OrderedDict()
            for i, (name, layer) in enumerate(self.backbone.named_children()):
                x = layer(x)
                if i in self.out_feats:
                    outputs[f"layer_{i}"] = x

            output = self.last_act(x)
            if self.num_classes > 0:
                output = torch.flatten(output, start_dim=1)
                output = self.fc(output)
            outputs["layer_last"] = output
            return outputs

    @staticmethod
    def make_plain_part(in_channels, hidden_channels, stride, n_blocks, pool_size, conv):
        layers = PlainNet.make_plain_part(in_channels, hidden_channels, 1, n_blocks, conv)
        if stride != 1:
            layers.append(nn.MaxPool2d(pool_size, stride=stride, padding=int(pool_size // 2)))
        return layers

    @staticmethod
    def make_backbone(n_blocks_list, stride_list, in_channels, hidden_channels, stride_factors, pool_size, conv):
        convs = []

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
            "fc_in_dim"      : None,
            "fc_out_dim"     : None,
            "num_classes"    : 10,
            "last_act"       : act,
            "conv"           : conv,
            "out_feats"      : None,
        }
        default_params = utils.set_params(default_params, configs, excluded_keys=["conv", "last_act"])
        return VGG(**default_params)