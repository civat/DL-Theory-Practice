import functools
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN

import register
from classification import utils


@register.NAME_TO_CONVS.register("Conv2d")
class Conv2d(nn.Module):
    """
    The meta-build block of neural network.
    This is a wrapper of `torch.nn.Conv2d`.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros", spectral_norm=False):
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : int or tuple of int
            The size of the kernel.
        stride : int or tuple of int
            The stride of the kernel.
        padding : int or tuple of int
            The padding of the input.
        dilation : int or tuple of int
            The dilation of the kernel.
        groups : int
            The number of groups.
        bias : bool
            Whether to use bias.
        padding_mode : str
            The padding mode.
        spectral_norm : bool
            Whether to use spectral normalization.
        """
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
            "stride"       : 1,
            "padding"      : 0,
            "dilation"     : 1,
            "groups"       : 1,
            "bias"         : True,
            "padding_mode" : "zeros",
            "spectral_norm": False,
        }

        # In most cases, we do not need to set the following parameters
        # as the network will infer them using stride and stride factor
        # of each layer.
        # The only reason we need to set them is when we want to use
        # 1x1 conv layers as fully connected layers.
        if "in_channels" in configs:
            default_params["in_channels"] = configs["in_channels"]
        if "out_channels" in configs:
            default_params["out_channels"] = configs["out_channels"]

        default_params = utils.set_params(default_params, configs)
        conv = functools.partial(Conv2d, **default_params)
        return conv


@register.NAME_TO_CONVS.register("Conv2dTranspose")
class Conv2dTranspose(nn.Module):
    """
    The meta-build block of neural network.
    This is a wrapper of `torch.nn.ConvTranspose2d`.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros", spectral_norm=False):
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : int or tuple of int
            The size of the kernel.
        stride : int or tuple of int
            The stride of the kernel.
        padding : int or tuple of int
            The padding of the input.
        dilation : int or tuple of int
            The dilation of the kernel.
        groups : int
            The number of groups.
        bias : bool
            Whether to use bias.
        padding_mode : str
            The padding mode.
        spectral_norm : bool
            Whether to use spectral normalization.
        """
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
    """
    The meta-build block of neural network.
    This is an upsample block consisting of `torch.nn.Upsample` and `torch.nn.Conv2d`.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros", mode="nearest", spectral_norm=False):
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : int or tuple of int
            The size of the kernel.
        stride : int or tuple of int
            The stride of the kernel.
        padding : int or tuple of int
            The padding of the input.
        dilation : int or tuple of int
            The dilation of the kernel.
        groups : int
            The number of groups.
        bias : bool
            Whether to use bias.
        padding_mode : str
            The padding mode.
        mode : str
            The upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        spectral_norm : bool
            Whether to use spectral normalization.
        """
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