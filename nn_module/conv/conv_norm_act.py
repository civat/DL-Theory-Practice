import functools
import torch.nn as nn

import register
from classification import utils
from nn_module.conv.convs import Conv2d
from nn_module.norm.dispatch_norm import DispatchNorm


@register.NAME_TO_CONVS.register("ConvNormAct")
class ConvNormAct(nn.Module):
    """
    Building block that is generally used in most convolutional neural networks.
    The ConvNormAct block is composed of one sub-block conv, one normalization layer and one activation layer.
    Example1:
        By setting conv to Conv2d, norm to BatchNorm2d and act to ReLU, the ConvNormAct block is simply a standard
        conv-bn-relu block.
    Example2:
        By setting conv to ResBlock(nn_module/conv/res_block/ResBlock), norm to BatchNorm2d and act to ReLU,
        the ConvNormAct block is a residual block defined in ResNet.
    """

    def __init__(self, in_channels, out_channels, stride, norm=nn.BatchNorm2d, act=nn.ReLU, dropout=0.0, pre_act=False, conv=Conv2d):
        super().__init__()
        if pre_act:
            self.convs = nn.Sequential(
                DispatchNorm(norm, num_features=in_channels),
                act(),
                conv(in_channels=in_channels, out_channels=out_channels, stride=stride),
                nn.Dropout(dropout)
            )
        else:
            self.convs = nn.Sequential(
                conv(in_channels=in_channels, out_channels=out_channels, stride=stride),
                DispatchNorm(norm, num_features=out_channels),
                act(),
                nn.Dropout(dropout)
            )

    def forward(self, x):
        return self.convs(x)

    @staticmethod
    def get_conv(configs):
        conv = register.get_conv(configs)
        norm = register.get_norm(configs["norm"])
        act = register.get_activation(configs["act"])

        default_params = {
            "norm"   : norm,
            "act"    : act,
            "dropout": 0.0,
            "pre_act": False,
            "conv"   : conv,
        }
        default_params = utils.set_params(default_params, configs, excluded_keys=["norm", "act", "conv"])
        conv = functools.partial(ConvNormAct, **default_params)
        return conv