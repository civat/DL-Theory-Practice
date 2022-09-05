import torch.nn as nn
import functools
from torch.nn.init import dirac_
from classification.models.dbb import conv_bn
from classification.models.dbb import transI_fusebn


class MobileOneBlockDP(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, padding_mode='zeros', deploy=False, r=1, conv_type="D"):
        assert conv_type in ["D", "P"]
        super(MobileOneBlockDP, self).__init__()
        self.deploy = deploy
        self.conv_type = conv_type

        # middle branch
        if conv_type == "D":
            k = kernel_size
            g = in_channels
            p = padding
            s = stride
        else:
            k = 1
            g = 1
            p = 0
            s = 1

        self.convs = nn.ModuleList([
            conv_bn(in_channels, out_channels, kernel_size=k, stride=s, padding=p, dilation=dilation,
                    groups=g, padding_mode=padding_mode) for _ in range(r)
        ])

        if conv_type == "D":
            # 1x1 branch
            self.conv_1x1 = conv_bn(in_channels, out_channels, (1, 1), stride, dilation=dilation, padding_mode=padding_mode)

        # BN branch
        self.delta = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, dilation=dilation,
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

            if self.conv_type == "D":
                output_1x1 = self.conv_1x1(x)
                return output_middle + output_1x1 + output_bn
            else:
                return output_middle + output_bn

        else:
            x = self.conv_fused(x)
            return x

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
        weight_m, biase_m = [], []
        for i in range(len(self.convs)):
            w, b = transI_fusebn(self.convs[i].conv.weight, self.convs[i].bn)
            weight_m.append(w)
            biase_m.append(b)
        weights_m = sum(weight_m)
        biases_m = sum(biase_m)

        if self.convs == "D":
            # fuse the 1x1 branch
            weight_1x1, bias_1x1 = transI_fusebn(self.conv_1x1.conv.weight, self.conv_1x1.bn)

        # fuse the BN branch
        weight_bn, bias_bn = transI_fusebn(self.delta.weight, self.bn)

        if self.convs == "D":
            weight = weights_m + weight_1x1 + weight_bn
            bias = biases_m + bias_1x1 + bias_bn
        else:
            weight = weights_m + weight_bn
            bias = biases_m + bias_bn
        self.conv_fused.weight.data = weight
        self.conv_fused.bias.data = bias

        self.__delattr__('convs')
        self.__delattr__('delta')
        self.__delattr__('bn')
        if self.conv_type == "D":
            self.__delattr__('conv_1x1')


class MobileOneBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=None, bias=None, padding_mode='zeros', deploy=False, r=1):
        super(MobileOneBlock, self).__init__()
        self.deploy = deploy
        self.conv_d = MobileOneBlockDP(in_channels, in_channels, kernel_size, stride, padding, dilation, padding_mode,
                                       r=r, conv_type="D")
        self.conv_p = MobileOneBlockDP(in_channels, out_channels, kernel_size, stride, padding, dilation, padding_mode,
                                       r=r, conv_type="P")

    def forward(self, x):
        x = self.conv_d(x)
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
