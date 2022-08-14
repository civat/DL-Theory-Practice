import torch
from torch import nn
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

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)
        self.init_params(out_channels)

    def forward(self, input):
        return F.conv2d(input, self.transform_weight(), self.bias, self.stride, self.padding, self.dilation)


class DiracGroup(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, norm, act, bias, n_layers):
        super().__init__()
        assert n_layers >= 1
        self.convs = [
            DiracConv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                        padding=int((kernel_size - 1) / 2)),
            norm(out_channels),
            act()
        ]
        for _ in range(1, n_layers):
            self.convs += [
                DiracConv2d(out_channels, out_channels, kernel_size=kernel_size, bias=bias,
                            padding=int((kernel_size - 1) / 2)),
                norm(out_channels),
                act()
            ]
        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        return self.convs(x)


@register.name_to_model.register("DiracNet")
class DiracNet(nn.Module):

    def __init__(self, in_channels, kernel_size_first, hidden_channels_first, stride_first, use_norm_first, 
                 use_act_first, kernel_size, hidden_channels_list, n_layers_list, norm, act, bias, num_classes):
        super().__init__()
        self.convs, out_channels = DiracNet.make_backbone(in_channels, kernel_size_first, hidden_channels_first, 
                                                          stride_first, use_norm_first, use_act_first, kernel_size, 
                                                          hidden_channels_list, n_layers_list, norm, act, bias)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        output = self.convs(x)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

    @staticmethod
    def make_backbone(in_channels, kernel_size_first, hidden_channels_first, stride_first, 
                      use_norm_first, use_act_first, kernel_size, hidden_channels_list, 
                      n_layers_list, norm, act, bias):
        p = int((kernel_size - 1) / 2)
        convs = [
            nn.Conv2d(in_channels, hidden_channels_first, kernel_size=kernel_size_first, stride=stride_first,
                      padding=p, bias=bias),
        ]
        if use_norm_first:
            convs.append(norm(hidden_channels_first))
        if use_act_first:
            convs.append(act())

        in_channels = hidden_channels_first
        for i, (hidden_channels, n_layers) in enumerate(zip(hidden_channels_list, n_layers_list)):
            convs.append(DiracGroup(in_channels, hidden_channels, kernel_size, norm, act, bias, n_layers))
            if i != (len(n_layers_list) - 1):
                convs.append(nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=p))
            in_channels = hidden_channels
            
        convs = nn.Sequential(*convs)
        return convs, hidden_channels

    @staticmethod
    def make_network(configs):
        norm = utils.get_norm(configs["norm"])
        act = utils.get_activation(configs["act"])

        default_params = {
            "in_channels": 3,
            "kernel_size_first": 3,
            "hidden_channels_first": 16,
            "stride_first": 1,
            "use_norm_first": True,
            "use_act_first": True,
            "kernel_size": 3,
            "hidden_channels_list": [64, 128, 256],
            "n_layers_list": [6, 6, 6],
            "norm": norm,
            "act": act,
            "bias": False,
            "num_classes": 10,
        }

        for key in default_params.keys():
            if key not in ["norm", "act"] and key in configs:
                default_params[key] = configs[key]

        return DiracNet(**default_params)
