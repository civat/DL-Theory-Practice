import functools
import torch
import torch.nn as nn

import register
from classification import utils
from nn_module.conv.convs import Conv2d
from nn_module.norm.dispatch_norm import DispatchNorm


@register.NAME_TO_CONVS.register("SqueezeConv")
class SqueezeConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros", spectral_norm=False, ratio=0.25, norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        hidden_channels = int(in_channels * ratio)
        self.sconv_1x1 = Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, bias=bias, spectral_norm=spectral_norm)
        self.norm = DispatchNorm(norm, num_features=hidden_channels)
        self.act = act()
        self.econv_1x1 = Conv2d(hidden_channels, int(out_channels/2), kernel_size=1, stride=1, dilation=dilation,
                               groups=groups, bias=bias, spectral_norm=spectral_norm)
        self.econv_kxk = Conv2d(hidden_channels, int(out_channels/2), kernel_size=kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, spectral_norm=spectral_norm)

    def forward(self, x):
        x = self.act(self.norm(self.sconv_1x1(x)))
        out1 = self.econv_1x1(x)
        out2 = self.econv_kxk(x)
        return torch.cat([out1, out1], dim=1)

    @staticmethod
    def get_conv(configs):
        norm = register.get_norm(configs["norm"])
        act = register.get_activation(configs["act"])
        default_params = {
            "kernel_size"  : 3,
            "stride"       : 1,
            "padding"      : 0,
            "dilation"     : 1,
            "groups"       : 1,
            "bias"         : True,
            "padding_mode" : "zeros",
            "spectral_norm": False,
            "ratio"        : 0.25,
            "norm"         : norm,
            "act"          : act,
        }
        default_params = utils.set_params(default_params, configs, excluded_keys=["norm", "act"])
        conv = functools.partial(SqueezeConv, **default_params)
        return conv


class SEConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride, ratio=0.25, act=nn.ReLU, conv=Conv2d):
        super().__init__()
        self.conv = conv(in_channels, out_channels, stride=stride)
        hidden_channels = int(out_channels * ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2d(out_channels, hidden_channels, kernel_size=1, stride=1),
            act(),
            Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.conv(x)
        scale = self.se(out)
        return out * scale

    @staticmethod
    def get_conv(configs):
        conv = register.get_conv(configs, "conv")
        act = register.get_activation(configs["act"])

        default_params = {
            "ratio": 0.25,
            "act"  : act,
            "conv" : conv,
        }
        default_params = utils.set_params(default_params, configs, excluded_keys=["conv", "act"])
        conv = functools.partial(SEConv, **default_params)
        return conv