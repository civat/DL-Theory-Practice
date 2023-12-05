import functools
import torch.nn as nn
import torch.nn.functional as F

import register
from classification import utils
from nn_module.conv.convs import Conv2d


@register.NAME_TO_CONVS.register("ResBlock")
class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, add=True, act=nn.ReLU, conv1=Conv2d, conv2=Conv2d, conv_sc=Conv2d):
        """Residual block introduced in ResNet
        Residual block consists of three convolutional layers.
          1) conv1: down-sampling or up-sampling layer.
          2) conv2: convolutional layer with stride 1.
          3) conv_sc: shortcut layer to match the size of the main stream tensor.

        The calculation of the residual block can be expressed as:
          1) output = conv2(conv1(x)) + conv_sc(x) if stride != 1
          2) output = conv2(conv1(x)) + x          if stride == 1

        We can use the "add" arg to set the operation of the residual block to
        subtraction:
          1) output = conv2(conv1(x)) - conv_sc(x) if stride != 1
          2) output = conv2(conv1(x)) - x          if stride == 1

        In general, it is meaningless to use subtraction in the residual block.
        See https://www.zhihu.com/question/433548556/answer/2938153423 for details
        of using subtraction instead of addition.

        Note that, we can simply make the residual block for up-sampling by
        setting "conv1" arg to ConvTranspose2d (nn_module/conv/convs/Conv2dTranspose).

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        stride : int or tuple
            The stride of the first convolutional layer (conv1).
        add : bool
            Whether to use addition in the residual block.
            "False" to use subtraction.
        act : nn.Module (see register.py for available activation functions)
            The activation function used after conv2.
        conv1 : nn.Module (see nn_module/conv/ for available convolutional layers)
            The first convolutional layer.
        conv2 : nn.Module (see nn_module/conv/ for available convolutional layers)
            The second convolutional layer.
        conv_sc : nn.Module (see nn_module/conv/ for available convolutional layers)
            The shortcut convolutional layer.
        """
        super().__init__()
        self.stride = stride
        self.add = add

        self.conv1 = conv1(in_channels, out_channels, stride=stride)
        self.conv2 = conv2(out_channels, out_channels, stride=1)

        self.shortcut = None

        # If shortcut tensor size is mismatched with the main stream tensor,
        # use 1x1 conv to match the size.
        if in_channels != out_channels or stride != 1:
            self.shortcut = conv_sc(in_channels, out_channels, stride=stride)
        else:
            self.shortcut = utils.Identity()
        self.act = act()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        skip = self.shortcut(x)
        if skip.size(-1) != x2.size(-1) or skip.size(-2) != x2.size(-2):
            skip = F.interpolate(skip, size=(x2.size(-2), x2.size(-1)))
        if self.add:
            out = x2 + skip
        else:
            out = x2 - skip
        return self.act(out)

    @staticmethod
    def get_conv(configs):
        conv1 = register.get_conv(configs, "conv1")
        conv2 = register.get_conv(configs, "conv2")
        conv_sc = register.get_conv(configs, "conv_sc")
        act = register.get_activation(configs["act"])

        default_params = {
            "add"    : True,
            "act"    : act,
            "conv1"  : conv1,
            "conv2"  : conv2,
            "conv_sc": conv_sc,
        }

        default_params = utils.set_params(default_params, configs, excluded_keys=["act", "conv1", "conv2", "conv_sc"])
        conv = functools.partial(ResBlock, **default_params)
        return conv