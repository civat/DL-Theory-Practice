import functools
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN

import register
from classification import utils


@register.NAME_TO_CONVS.register("Conv2d")
class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros", spectral_norm=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        if spectral_norm:
            self.conv = SN(self.conv)

    def forward(self, x):
        return self.conv(x)

    @staticmethod
    def get_conv(configs):
        default_params = {
            "kernel_size"  : 3,
            "padding"      : 0,
            "dilation"     : 1,
            "groups"       : 1,
            "bias"         : True,
            "padding_mode" : "zeros",
            "spectral_norm": False,
        }
        default_params = utils.set_params(default_params, configs)
        conv = functools.partial(Conv2d, **default_params)
        return conv


@register.NAME_TO_CONVS.register("Conv2dTranspose")
class Conv2dTranspose(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros", spectral_norm=False):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, padding_mode=padding_mode)
        if spectral_norm:
            self.conv = SN(self.conv)

    def forward(self, x):
        return self.conv(x)

    @staticmethod
    def get_conv(configs):
        default_params = {
            "kernel_size"  : 3,
            "padding"      : 0,
            "dilation"     : 1,
            "groups"       : 1,
            "bias"         : True,
            "padding_mode" : "zeros",
            "spectral_norm": False,
        }
        default_params = utils.set_params(default_params, configs)
        conv = functools.partial(Conv2dTranspose, **default_params)
        return conv


@register.NAME_TO_CONVS.register("Conv2dUp")
class Conv2dUp(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros", mode="nearest", spectral_norm=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=stride, mode=mode)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        if spectral_norm:
            self.conv = SN(self.conv)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

    @staticmethod
    def get_conv(configs):
        default_params = {
            "kernel_size"  : 3,
            "padding"      : 0,
            "dilation"     : 1,
            "groups"       : 1,
            "bias"         : True,
            "padding_mode" : "zeros",
            "mode"         : "nearest",
            "spectral_norm": False,
        }
        default_params = utils.set_params(default_params, configs)
        conv = functools.partial(Conv2dUp, **default_params)
        return conv


def set_sepectral_norm(convs):
    if isinstance(convs, Conv2d) or isinstance(convs, Conv2dTranspose):
        return SN(convs)

    for i, module in enumerate(convs):
        if isinstance(module, Conv2d) or isinstance(module, Conv2dTranspose):
            convs[i] = SN(module)
    return convs