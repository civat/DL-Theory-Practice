import functools
import torch.nn as nn

import register
from classification import utils
from nn_module.conv.convs import Conv2d
from nn_module.norm.dispatch_norm import DispatchNorm


@register.NAME_TO_CONVS.register("DSConv")
class DSConv(nn.Module):
    """
    Depthwise Separable Convolution used in MobileNetV1.
    The depthwise convolution is responsible for shape transformation.
    The pointwise convolution is responsible for channel transformation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, bias=True,
                 padding_mode="zeros", spectral_norm=False, norm=nn.BatchNorm2d, act=nn.ReLU):
        """
        Note: the args "norm" and "act" are used to specify normalization and activation
        between depthwise convolution and pointwise convolution.

        Parameters
        ----------
        in_channels: int
            The number of input channels.
        out_channels: int
            The number of output channels.
        kernel_size: int or tuple of int
            The size of the kernel.
        stride: int or tuple of int
            The stride of the kernel.
        padding: int or tuple of int
            The padding of the input.
        dilation: int or tuple of int
            The dilation of the kernel.
        bias: bool
            Whether to use bias.
        padding_mode: str
            The padding mode.
        spectral_norm: bool
            Whether to use spectral normalization.
        norm: nn.Module
            Normalization used between depthwise
            convolution and pointwise convolution.
        act: nn.Module
            Activation used between depthwise
            convolution and pointwise convolution.
        """
        super().__init__()
        self.convs = []
        conv_d = Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                        groups=in_channels, bias=bias, padding_mode=padding_mode, spectral_norm=spectral_norm)
        self.convs += [conv_d]
        self.convs += [
            DispatchNorm(norm, num_features=in_channels),
            act(),
        ]
        conv_p = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=dilation,
                        groups=1, bias=bias, padding_mode=padding_mode, spectral_norm=spectral_norm)
        self.convs += [conv_p]
        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        return self.convs(x)

    @staticmethod
    def get_conv(configs):
        norm = register.get_norm(configs["norm"])
        act = register.get_activation(configs["act"])
        default_params = {
            "kernel_size"  : 3,
            "padding"      : 0,
            "dilation"     : 1,
            "bias"         : True,
            "padding_mode" : "zeros",
            "spectral_norm": False,
            "norm"         : norm,
            "act"          : act,
        }
        default_params = utils.set_params(default_params, configs, excluded_keys=["norm", "act"])
        conv = functools.partial(DSConv, **default_params)
        return conv


@register.NAME_TO_CONVS.register("DSBottleNeckConv")
class DSBottleNeckConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, bias=True,
                 padding_mode="zeros", spectral_norm=False, expansion=1, norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        hidden_channels = int(in_channels * expansion)

        # if expansion is 1, do not add the first 1x1 conv
        if expansion == 1:
            convs = [
                Conv2d(in_channels, hidden_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                       groups=hidden_channels, bias=bias, padding_mode=padding_mode, spectral_norm=spectral_norm),
                DispatchNorm(norm, num_features=hidden_channels),
                act(),
                Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=dilation,
                       groups=1, bias=bias, padding_mode=padding_mode, spectral_norm=spectral_norm)
            ]
        else:
            convs = [
                Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, dilation=dilation,
                       groups=1, bias=bias, padding_mode=padding_mode, spectral_norm=spectral_norm),
                DispatchNorm(norm, num_features=hidden_channels),
                act(),
                Conv2d(hidden_channels, hidden_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                       groups=hidden_channels, bias=bias, padding_mode=padding_mode, spectral_norm=spectral_norm),
                DispatchNorm(norm, num_features=hidden_channels),
                act(),
                Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=dilation,
                       groups=1, bias=bias, padding_mode=padding_mode, spectral_norm=spectral_norm)
            ]

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        out = self.convs(x)
        # add residual connection if the
        # input and output have the same shape
        if x.size(1) == out.size(1) and x.size(2) == out.size(2) and x.size(3) == out.size(3):
            out += x
        return out

    @staticmethod
    def get_conv(configs):
        norm = register.get_norm(configs["norm"])
        act = register.get_activation(configs["act"])
        default_params = {
            "kernel_size"  : 3,
            "padding"      : 0,
            "dilation"     : 1,
            "bias"         : True,
            "padding_mode" : "zeros",
            "spectral_norm": False,
            "expansion"    : 1,
            "norm"         : norm,
            "act"          : act,
        }
        default_params = utils.set_params(default_params, configs, excluded_keys=["norm", "act"])
        conv = functools.partial(DSBottleNeckConv, **default_params)
        return conv