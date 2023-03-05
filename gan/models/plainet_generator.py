import torch.nn as nn

import register
from classification import utils


@register.name_to_model.register("PlainetGenVec")
class PlainetGenVec(nn.Module):
    """Plain net based generator."""

    def __init__(self, input_dim, hidden_channels, hidden_size, out_channels, kernel_size_up, kernel_size_eq,
                 stride, bias, norm, act, dropout, n_layers, output_conv):
        super(PlainetGenVec, self).__init__()
        self.hidden_channels = hidden_channels
        if isinstance(hidden_size, int):
            w = h = hidden_size
        elif isinstance(hidden_size, tuple):
            w, h = hidden_size
        self.w, self.h = w, h

        self.fc = nn.Linear(input_dim, w * h * hidden_channels)
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.act = act()

        self.decoder = PlainetGenVec.make_decoder(hidden_channels, out_channels, kernel_size_up, kernel_size_eq,
                                                  stride, bias, norm, act, dropout, n_layers, output_conv)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.hidden_channels, self.w, self.h)
        x = self.act(self.bn(x))
        x = self.decoder(x)
        return x

    @staticmethod
    def make_decoder(in_channels, out_channels, kernel_size_up, kernel_size_eq, stride, bias, norm, act, dropout, n_layers, output_conv):
        convs = []
        p = int((kernel_size_up - 1) / 2)

        for _ in range(n_layers - 1):
            hidden_channels = int(in_channels / stride)
            convs += [
                nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=kernel_size_up, stride=stride, bias=bias, padding=p),
                norm(hidden_channels),
                act(),
                nn.Dropout(dropout),
            ]
            in_channels = int(in_channels / stride)
        if output_conv:
            hidden_channels = int(in_channels / stride)
            convs += [
                nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=kernel_size_up, stride=stride, bias=bias, padding=p),
                norm(hidden_channels),
                act(),
                nn.Dropout(dropout),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size_eq, stride=1, bias=bias, padding=int(kernel_size_eq // 2))
            ]
        else:
            convs += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size_up, stride=stride, bias=bias, padding=p),
            ]

        convs += [nn.Tanh()]
        convs = nn.Sequential(*convs)
        return convs

    @staticmethod
    def make_network(configs):
        norm = utils.get_norm(configs["norm"])
        act = utils.get_activation(configs["act"])

        default_params = {
            "input_dim"      : 128,
            "hidden_channels": 512,
            "hidden_size"    : 6,
            "out_channels"   : 3,
            "kernel_size_up" : 4,
            "kernel_size_eq" : 3,
            "stride"         : 2,
            "norm"           : norm,
            "act"            : act,
            "bias"           : False,
            "dropout"        : 0,
            "n_layers"       : 3,
            "output_conv"    : False
        }

        default_params = utils.set_params(default_params, configs, excluded_keys=["norm", "act"])
        return PlainetGenVec(**default_params)