import functools
import torch.nn as nn

import register
from classification import utils
from nn_module.conv.convs import Conv2d


@register.NAME_TO_CONVS.register("ConvDis")
class ConvDis(nn.Module):

    def __init__(self, in_channels, out_channels, stride, conv1=Conv2d, conv2=Conv2d):
        super().__init__()
        self.conv1 = conv1(in_channels=in_channels, out_channels=out_channels, stride=1)
        self.conv2 = conv2(in_channels=out_channels, out_channels=out_channels, stride=stride)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x2

    @staticmethod
    def get_conv(configs):
        conv1 = register.get_conv(configs, "conv1")
        conv2 = register.get_conv(configs, "conv2")

        default_params = {
            "conv1": conv1,
            "conv2": conv2,
        }
        default_params = utils.set_params(default_params, configs, excluded_keys=["conv1", "conv2"])
        conv = functools.partial(ConvDis, **default_params)
        return conv