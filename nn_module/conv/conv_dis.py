import functools
import torch.nn as nn

import register
from classification import utils
from nn_module.conv.convs import Conv2d


@register.NAME_TO_CONVS.register("ConvDis")
class ConvDis(nn.Module):
    """
    Building block that is generally used in GAN as blocks of discriminator.
    The ConvDis block is composed of two sub-blocks: conv1 and conv2.
    The conv1 is with stride 1 and conv2 is with stride specified in args.
    When set conv1 and conv2 to standard Conv2d, the ConvDis block is equivalent to
    the block of discriminator used in SN-GAN:
    https://arxiv.org/pdf/1802.05957.pdf
    """

    def __init__(self, in_channels, out_channels, stride, conv1=Conv2d, conv2=Conv2d):
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        stride : int or tuple of int
            The stride of the kernel.
        conv1 : nn.Module (any block defined in nn_module\conv)
            The first building block.
        conv2 : nn.Module (any block defined in nn_module\conv)
            The second building block.
        """
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