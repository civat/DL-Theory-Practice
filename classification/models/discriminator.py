import torch.nn as nn

import register
from classification import utils
from classification.models.plainnet import PlainNet
from nn_module.conv.convs import Conv2d


@register.name_to_model.register("SimpleDiscriminator")
class SimpleDiscriminator(nn.Module):
    """
    Implements the discriminator used in SNGAN:
    https://arxiv.org/pdf/1802.05957.pdf
    """

    def __init__(self, in_channels, hidden_channels, stride_list, stride_factor, num_classes, conv1=Conv2d, conv2=Conv2d):
        super().__init__()
        assert len(stride_list) > 0
        self.backbone = [conv1(in_channels, hidden_channels, stride=stride_list[0])]
        stride_factors = [PlainNet._get_stride_factor(stride, stride_factor) for stride in stride_list]

        for i in range(1, len(stride_factors)):
            in_channels = hidden_channels
            hidden_channels = hidden_channels * stride_factors[i]
            self.backbone += [conv1(in_channels, hidden_channels, stride=stride_list[i])]

        conv_last = conv2(hidden_channels, hidden_channels * 2, stride=1)
        self.backbone += [conv_last]
        self.backbone = nn.Sequential(*self.backbone)
        if num_classes == 2:
            num_classes = 1
        if num_classes > 0:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(hidden_channels * 2, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        output = self.backbone(x)
        if self.num_classes > 0:
            output = self.avg_pool(output)
            output = output.view(output.size(0), -1)
            output = self.fc(output)
        return output

    @staticmethod
    def make_network(configs):
        conv1 = register.get_conv(configs, "conv1")
        conv2 = register.get_conv(configs, "conv2")

        default_params = {
            "in_channels"    : 3,
            "hidden_channels": 64,
            "stride_list"    : [2, 2, 2],
            "stride_factor"  : 2,
            "num_classes"    : 2,
            "conv1"          : conv1,
            "conv2"          : conv2,
        }
        default_params = utils.set_params(default_params, configs, excluded_keys=["conv1", "conv2"])
        return SimpleDiscriminator(**default_params)