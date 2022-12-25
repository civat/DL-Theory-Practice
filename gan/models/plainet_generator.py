import torch.nn as nn

import register
from classification.utils import get_norm
from classification.utils import get_activation


@register.name_to_model.register("PlainetGenVec")
class PlainetGenVec(nn.Module):
    """Plain net based generator."""

    def __init__(self, input_dim, hidden_channels, hidden_size, out_channels, kernel_size, stride, bias, norm, act, dropout,
                 n_layers, output_kernel_size, output_padding_mode):
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

        self.decoder = PlainetGenVec.make_decoder(hidden_channels, kernel_size, stride, bias, norm, act, dropout, n_layers)
        p = int((output_kernel_size - 1) / 2)
        hidden_channels = int(hidden_channels / (stride ** n_layers))
        self.conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=output_kernel_size, stride=1, bias=bias, padding=p, padding_mode=output_padding_mode)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.hidden_channels, self.w, self.h)
        x = self.act(self.bn(x))
        x = self.decoder(x)
        x = self.conv(x)
        x = self.tanh(x)
        return x

    @staticmethod
    def make_decoder(in_channels, kernel_size, stride, bias, norm, act, dropout, n_layers):
        convs = []
        p = int((kernel_size - 1) / 2)

        for _ in range(n_layers):
            hidden_channels = int(in_channels / stride)
            convs += [
                nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=kernel_size, stride=stride, bias=bias, padding=p),
                norm(hidden_channels),
                act(),
                nn.Dropout(dropout),
            ]
            in_channels = int(in_channels / stride)

        convs = nn.Sequential(*convs)
        return convs

    @staticmethod
    def make_network(configs):
        norm = get_norm(configs["norm"])
        act = get_activation(configs["act"])

        default_params = {
            "input_dim": 128,
            "hidden_channels": 512,
            "hidden_size": 6,
            "out_channels": 3,
            "kernel_size": 3,
            "stride": 2,
            "norm": norm,
            "act": act,
            "bias": False,
            "dropout": 0,
            "n_layers": 3,
            "output_kernel_size": 4,
            "output_padding_mode": "reflect",
        }

        for key in default_params.keys():
            if key not in ["norm", "act"] and key in configs:
                default_params[key] = configs[key]

        return PlainetGenVec(**default_params)
