import torch.nn as nn

import register


@register.NAME_TO_CONVS.register("Conv2d")
class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros", deploy=False):
        super().__init__()
        self.deploy = deploy
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        return self.conv(x)

    def switch_to_deploy(self):
        self.deploy = True
        pass

    def get_params(self):
        return self.conv.weight.data, self.conv.bias.data

    @staticmethod
    def get_conv(configs):
        return Conv2d