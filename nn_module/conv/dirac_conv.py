import warnings
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import dirac_

import register
from classification import utils


def normalize(w):
    """Normalizes weight tensor over full filter."""
    return F.normalize(w.view(w.shape[0], -1)).view_as(w)


class DiracConv(nn.Module):

    def init_params(self, out_channels):
        self.alpha = nn.Parameter(torch.Tensor(out_channels).fill_(1))
        self.beta = nn.Parameter(torch.Tensor(out_channels).fill_(0.1))
        self.register_buffer('delta', dirac_(self.weight.data.clone()))
        assert self.delta.shape == self.weight.shape
        self.v = (-1,) + (1,) * (self.weight.dim() - 1)

    def transform_weight(self):
        return self.alpha.view(*self.v) * self.delta + self.beta.view(*self.v) * normalize(self.weight)


@register.NAME_TO_CONVS.register("DiracConv2d")
class DiracConv2d(nn.Conv2d, DiracConv):
    """Dirac parametrized convolutional layer.
    Works the same way as `nn.Conv2d`, but has additional weight parametrizatoin:
        :math:`\alpha\delta + \beta W`,
    where:
        :math:`\alpha` and :math:`\beta` are learnable scalars,
        :math:`\delta` is such a tensor so that `F.conv2d(x, delta) = x`, ie
            Kroneker delta
        `W` is weight tensor
    It is user's responsibility to set correcting padding. Only stride=1 supported.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        if stride != 1:
            warnings.warn("DiracConv2d only supports stride=1 ! For case stride != 1, DiracConv2d still uses stride=1."
                          "So users need to ensure the correctness of the network when stride != 1 is specified.")
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)

    def forward(self, input):
        return F.conv2d(input, self.transform_weight(), self.bias, self.stride, self.padding, self.dilation)

    @staticmethod
    def get_conv(configs):
        default_params = {
            "kernel_size" : 3,
            "padding"     : 0,
            "dilation"    : 1,
            "groups"      : 1,
            "bias"        : True,
            "padding_mode": "zeros",
        }
        default_params = utils.set_params(default_params, configs)
        conv = functools.partial(DiracConv2d, **default_params)
        return conv