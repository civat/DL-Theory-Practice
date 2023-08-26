import torch.nn as nn

import register
from classification import utils
from nn_module.conv.convs import Conv2d


@register.name_to_model.register("PlainNet")
class PlainNet(nn.Module):

    def __init__(self, in_channels, hidden_channels, n_blocks_list, stride_list, stride_factor, num_classes, last_act, conv=Conv2d):
        super().__init__()
        assert len(n_blocks_list) > 0
        assert len(n_blocks_list) == len(stride_list)
        self.convs, out_channels = self.make_backbone(n_blocks_list, stride_list, in_channels, hidden_channels,
                                                      stride_factor, conv)
        self.last_act = last_act()

        # For binary classification task, we use BCE loss so only one output logit is needed.
        self.num_classes = num_classes
        if num_classes == 2:
            num_classes = 1
        if num_classes > 0:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        output = self.last_act(self.convs(x))
        if self.num_classes > 0:
            output = self.avg_pool(output)
            output = output.view(output.size(0), -1)
            output = self.fc(output)
        return output

    @staticmethod
    def make_plain_part(in_channels, hidden_channels, stride, n_blocks, conv):
        # The first block is with the specified "stride", and all others are with stride=1.
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(conv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                stride=stride,
                ))
            in_channels = hidden_channels
        return layers

    @staticmethod
    def make_backbone(n_blocks_list, stride_list, in_channels, hidden_channels, stride_factor, conv):
        convs = []
        stride_factors = [PlainNet._get_stride_factor(stride, stride_factor) for stride in stride_list]

        for i, (n_blocks, stride, stride_factor) in enumerate(zip(n_blocks_list, stride_list, stride_factors)):
            if i != 0:
                hidden_channels = int(in_channels * stride_factor)
            convs += PlainNet.make_plain_part(in_channels, hidden_channels, stride, n_blocks, conv)
            in_channels = hidden_channels

        convs = nn.Sequential(*convs)
        return convs, hidden_channels

    @staticmethod
    def _get_stride_factor(stride, stride_factor):
        return 1 if stride == 1 else stride_factor

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
            "conv"           : conv,
        }
        default_params = utils.set_params(default_params, configs, excluded_keys=["conv", "last_act"])
        return PlainNet(**default_params)


@register.name_to_model.register("PlainNet18")
class PlainOANet(nn.Module):

    def __init__(self, in_channels, hidden_channels, n_blocks_list, stride_list, stride_factor, num_classes, last_act,
                 use_pool, conv1=Conv2d, conv2=Conv2d):
        super().__init__()
        assert len(n_blocks_list) >= 2
        assert len(n_blocks_list) == len(stride_list)
        self.convs = PlainNet.make_plain_part(in_channels, hidden_channels, stride_list[0], n_blocks_list[0], conv1)
        if use_pool:
            self.convs += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        self.convs = nn.Sequential(*self.convs)

        in_channels = hidden_channels
        hidden_channels = in_channels * stride_factor if stride_list[0] != 1 else in_channels
        self.plainnet = PlainNet(in_channels, hidden_channels, n_blocks_list[1:], stride_list[1:], stride_factor,
                                 num_classes, last_act, conv=conv2)

    def forward(self, x):
        x1 = self.convs(x)
        x2 = self.plainnet(x1)
        return x2

    @staticmethod
    def make_network(configs):
        conv1 = register.get_conv(configs, "conv1")
        conv2 = register.get_conv(configs, "conv2")
        act = register.get_activation(configs["last_act"])

        default_params = {
            "in_channels"    : 3,
            "hidden_channels": 64,
            "n_blocks_list"  : [2, 2, 2, 2],
            "stride_list"    : [2, 1, 1, 1],
            "stride_factor"  : 2,
            "num_classes"    : 10,
            "last_act"       : act,
            "use_pool"       : True,
            "conv1"          : conv1,
            "conv2"          : conv2,
        }
        default_params = utils.set_params(default_params, configs, excluded_keys=["conv1", "conv2", "last_act"])
        return PlainOANet(**default_params)