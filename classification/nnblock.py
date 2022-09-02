import functools

import torch.nn as nn


class ConvGroup(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=False,
                 padding_mode="zeros", conv=None, deploy=False, k=1):
        super(ConvGroup, self).__init__()
        self.deploy = deploy
        self.convs = nn.ModuleList([
            conv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
            for _ in range(k)
        ])
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

    def forward(self, x):
        if not self.deploy:
            output = [conv(x) for conv in self.convs]
            output = sum(output)
        else:
            output = self.fused_conv(x)
        return output

    def switch_to_deploy(self):
        self.deploy = True
        self.fused_conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                                    self.padding, self.dilation, self.groups, self.bias, self.padding_mode)
        for conv in self.convs:
            conv.switch_to_deploy()
            param, bias = conv.get_param()
            self.fused_conv.weight.data += param
            self.fused_conv.bias += bias
        self.__delattr__('convs')


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
        k = configs["k"] if "k" in configs.keys() else 1
        conv_group = functools.partial(ConvGroup, conv=Conv2d, k=k)
        return conv_group

