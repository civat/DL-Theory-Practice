import torch.nn as nn

import register
from classification import utils
from classification import nnblock
from classification.models import tools


class PlainBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, norm, act, bias, dropout=.0, conv=nnblock.Conv2d):
        super().__init__()
        p = int((kernel_size - 1) / 2)
        self.stride = stride
        self.convs = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=p, bias=bias),
            norm(out_channels),
            act(),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.convs(x)


@register.name_to_model.register("PlainNet")
class PlainNet(nn.Module):

    def __init__(self, block, n_blocks_list, stride_list, in_channels, hidden_channels, kernel_size,
                 norm, act, bias, num_classes, dropout, conv=nnblock.Conv2d):
        super().__init__()
        self.convs, out_channels = PlainNet.make_backbone(block, n_blocks_list, stride_list, in_channels, hidden_channels,
                                                          kernel_size, norm, act, bias, dropout, conv)
        # For binary classification task, we use BCE loss so only one output logit is needed.
        self.num_classes = num_classes
        if num_classes == 2:
            num_classes = 1
        if num_classes > 0:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        output = self.convs(x)
        if self.num_classes > 0:
            output = self.avg_pool(output)
            output = output.view(output.size(0), -1)
            output = self.fc(output)
        return output

    @staticmethod
    def _make_plain_part(block, in_channels, hidden_channels, kernel_size, stride, norm, act, bias, n_blocks, dropout, conv=nnblock.Conv2d):
        # The first block is with the specified "stride", and all others are with stride=1.
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(in_channels, hidden_channels, kernel_size, stride, norm, act, bias, dropout, conv))
            in_channels = hidden_channels
        return layers

    @staticmethod
    def make_backbone(block, n_blocks_list, stride_list, in_channels, hidden_channels, kernel_size,
                      norm, act, bias, dropout, conv=nnblock.Conv2d):
        convs = []
        for i, (n_blocks, stride) in enumerate(zip(n_blocks_list, stride_list)):
            if i != 0:
                hidden_channels = in_channels * stride
            convs += PlainNet._make_plain_part(block, in_channels, hidden_channels, kernel_size, stride, norm, act, bias, n_blocks, dropout, conv)
            in_channels = hidden_channels

        convs = nn.Sequential(*convs)
        return convs, hidden_channels

    @staticmethod
    def make_network(configs):
        conv = tools.get_conv(configs)
        norm = utils.get_norm(configs["norm"])
        act = utils.get_activation(configs["act"])
        block = PlainBlock

        default_params = {
            "conv": conv,
            "block": block,
            "n_blocks_list": [2, 2, 2, 2],
            "stride_list": [2, 1, 1, 1],
            "in_channels": 3,
            "hidden_channels": 64,
            "kernel_size": 3,
            "norm": norm,
            "act": act,
            "bias": False,
            "num_classes": 10,
            "dropout": 0,
        }

        default_params = utils.set_params(default_params, configs, excluded_keys=["conv", "block", "norm", "act"])
        return PlainNet(**default_params)