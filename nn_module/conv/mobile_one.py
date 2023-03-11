import functools
import torch.nn as nn
from torch.nn.init import dirac_

import register
from nn_module.conv.dbb import conv_bn
from nn_module.conv.dbb import transI_fusebn
from nn_module.conv.dbb import transVI_multiscale


class MobileOneBlockD(nn.Module):

    def __init__(self, in_channels, kernel_size,
                 stride=1, padding=0, dilation=1, padding_mode='zeros', deploy=False, r=1):
        super(MobileOneBlockD, self).__init__()
        self.deploy = deploy

        # middle branch
        self.convs = nn.ModuleList([
            conv_bn(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation,
                    groups=in_channels, padding_mode=padding_mode) for _ in range(r)
        ])

        # 1x1 branch
        self.conv_1x1 = conv_bn(in_channels, in_channels, (1, 1), stride, dilation=dilation, groups=in_channels,
                                padding_mode=padding_mode)

        # BN branch
        self.delta = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation,
                               groups=in_channels, padding_mode=padding_mode, bias=False)
        dirac_(self.delta.weight)
        self.delta.requires_grad_(False)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        if not self.deploy:
            output_middle = [conv(x) for conv in self.convs]
            output_middle = sum(output_middle)
            output_bn = self.delta(x)
            output_bn = self.bn(output_bn)
            output_1x1 = self.conv_1x1(x)
            return output_middle + output_1x1 + output_bn
        else:
            return self.conv_fused(x)

    def switch_to_deploy(self):
        if self.deploy:
            return
        self.deploy = True
        conv = self.convs[0].conv
        self.conv_fused = nn.Conv2d(in_channels=conv.in_channels,
                                    out_channels=conv.out_channels,
                                    kernel_size=conv.kernel_size,
                                    stride=conv.stride,
                                    padding=conv.padding,
                                    dilation=conv.dilation,
                                    groups=conv.groups,
                                    bias=True)

        # fuse middle branch
        # first fuse each Conv and BN
        weight_m, bias_m = [], []
        for i in range(len(self.convs)):
            w, b = transI_fusebn(self.convs[i].conv.weight, self.convs[i].bn)
            weight_m.append(w)
            bias_m.append(b)
        weight_m = sum(weight_m)
        bias_m = sum(bias_m)

        # fuse the 1x1 branch
        weight_1x1, bias_1x1 = transI_fusebn(self.conv_1x1.conv.weight, self.conv_1x1.bn)
        weight_1x1 = transVI_multiscale(weight_1x1, conv.kernel_size[0])

        # fuse the BN branch
        weight_bn, bias_bn = transI_fusebn(self.delta.weight, self.bn)

        weight = weight_m + weight_1x1 + weight_bn
        bias = bias_m + bias_1x1 + bias_bn

        self.conv_fused.weight.data = weight
        self.conv_fused.bias.data = bias

        self.__delattr__('convs')
        self.__delattr__('delta')
        self.__delattr__('bn')
        self.__delattr__('conv_1x1')


class MobileOneBlockP(nn.Module):

    def __init__(self, in_channels, out_channels, dilation=1, padding_mode='zeros', deploy=False, r=1):
        super(MobileOneBlockP, self).__init__()
        self.deploy = deploy

        self.convs = nn.ModuleList([
            conv_bn(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=dilation,
                    padding_mode=padding_mode) for _ in range(r)
        ])

        # BN branch
        self.delta = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=dilation,
                               padding_mode=padding_mode, bias=False)
        dirac_(self.delta.weight)
        self.delta.requires_grad_(False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if not self.deploy:
            output_middle = [conv(x) for conv in self.convs]
            output_middle = sum(output_middle)
            output_bn = self.delta(x)
            output_bn = self.bn(output_bn)
            return output_middle + output_bn
        else:
            return self.conv_fused(x)

    def switch_to_deploy(self):
        if self.deploy:
            return
        self.deploy = True
        conv = self.convs[0].conv
        self.conv_fused = nn.Conv2d(in_channels=conv.in_channels,
                                    out_channels=conv.out_channels,
                                    kernel_size=conv.kernel_size,
                                    stride=conv.stride,
                                    padding=conv.padding,
                                    dilation=conv.dilation,
                                    bias=True)

        # fuse middle branch
        # first fuse each Conv and BN
        weight_m, bias_m = [], []
        for i in range(len(self.convs)):
            w, b = transI_fusebn(self.convs[i].conv.weight, self.convs[i].bn)
            weight_m.append(w)
            bias_m.append(b)
        weight_m = sum(weight_m)
        bias_m = sum(bias_m)

        # fuse the BN branch
        weight_bn, bias_bn = transI_fusebn(self.delta.weight, self.bn)

        weight = weight_m + weight_bn
        bias = bias_m + bias_bn
        self.conv_fused.weight.data = weight
        self.conv_fused.bias.data = bias

        self.__delattr__('convs')
        self.__delattr__('delta')
        self.__delattr__('bn')


@register.NAME_TO_CONVS.register("MobileOneBlock")
class MobileOneBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=None, bias=None, padding_mode='zeros', deploy=False, r=1):
        super(MobileOneBlock, self).__init__()
        self.deploy = deploy
        self.conv_d = MobileOneBlockD(in_channels, kernel_size, stride, padding, dilation, padding_mode, deploy=False, r=r)
        self.conv_p = MobileOneBlockP(in_channels, out_channels, dilation, padding_mode, deploy=False, r=r)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv_d(x))
        x = self.conv_p(x)
        return x

    def switch_to_deploy(self):
        self.conv_d.switch_to_deploy()
        self.conv_p.switch_to_deploy()

    def get_params(self):
        raise Exception("Not supported by the MobileOneBlock!")

    @staticmethod
    def get_conv(configs):
        k = configs["k"] if "k" in configs.keys() else 1
        default_params = {
            "r": 1,
        }
        for key in default_params.keys():
            if key in configs:
                default_params[key] = configs[key]

        conv = functools.partial(MobileOneBlock, **default_params)
        return conv
