import torch
import functools
import torch.nn as nn

import register
from classification import utils


class InceptionBlock(nn.Module):
    """
    Building block of Inception networks.
    """

    def __init__(self, in_channels, stride, out_channels_1x1, reduction_3x3, out_channels_3x3, reduction_5x5,
                 out_channels_5x5, factorize_5x5, reduction_pool, norm, act, bias, pool):
        super(InceptionBlock, self).__init__()
        self.conv1x1 = None
        if out_channels_1x1 != 0:
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels_1x1, kernel_size=1, stride=stride, bias=bias),
                norm(out_channels_1x1),
                act(),
            )
        if reduction_3x3 != 0:
            self.conv3x3 = nn.Sequential(
                nn.Conv2d(in_channels, reduction_3x3, kernel_size=1, bias=bias),
                norm(reduction_3x3),
                act(),
                nn.Conv2d(reduction_3x3, out_channels_3x3, kernel_size=3, stride=stride, padding=1, bias=bias),
                norm(out_channels_3x3),
                act(),
            )
        else:
            self.conv3x3 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels_3x3, kernel_size=3, stride=stride, padding=1, bias=bias),
                norm(out_channels_3x3),
                act(),
            )

        self.conv5x5 = []
        if reduction_5x5 != 0:
            self.conv5x5 += [
                nn.Conv2d(in_channels, reduction_5x5, kernel_size=1, bias=bias),
                norm(reduction_5x5),
                act(),
            ]
        if factorize_5x5:
            self.conv5x5 += [
                nn.Conv2d(reduction_5x5, out_channels_5x5, kernel_size=3, stride=stride, padding=1, bias=bias),
                norm(out_channels_5x5),
                act(),
                nn.Conv2d(out_channels_5x5, out_channels_5x5, kernel_size=3, stride=1, padding=1, bias=bias),
                norm(out_channels_5x5),
                act(),
            ]
        else:
            self.conv5x5 += [
                nn.Conv2d(reduction_5x5, out_channels_5x5, kernel_size=5, stride=stride, padding=2, bias=bias),
                norm(out_channels_5x5),
                act(),
            ]
        self.conv5x5 = nn.Sequential(*self.conv5x5)

        if reduction_pool != 0:
            self.conv1x1_pool = nn.Sequential(
                pool(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_channels, reduction_pool, kernel_size=1, bias=bias),
                norm(reduction_pool),
                act(),
            )
        else:
            self.conv1x1_pool = nn.Sequential(
                pool(kernel_size=3, stride=stride, padding=1)
            )

    def forward(self, x):
        out = []
        if self.conv1x1 is not None:
            out.append(self.conv1x1(x))
        out.append(self.conv3x3(x))
        out.append(self.conv5x5(x))
        out.append(self.conv1x1_pool(x))
        out = torch.concat(out, dim=1)
        return out


@register.name_to_model.register("Inception")
class Inception(nn.Module):
    """
    Implementation of Inception networks.
    See following papers for details:
        1. Inception v1: "Going Deeper with Convolutions";
        2. Inception v2: "Batch Normalization: Accelerating 
           Deep Network Training by Reducing Internal Covariate Shift";
    """

    def __init__(self, in_channels, base_channels, kernel_size_first, stride_first, maxpool_first, norm, act, bias,
                 dropout, factorize_5x5, num_classes):
        super(Inception, self).__init__()
        self.convs, out_channels = Inception.make_backbone(in_channels, base_channels, kernel_size_first, stride_first,
                                                           maxpool_first, norm, act, bias, factorize_5x5)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        output = self.convs(x)
        output = self.dropout(self.avg_pool(output))
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

    @staticmethod
    def make_backbone(in_channels, base_channels, kernel_size_first, stride_first, maxpool_first, norm, act, bias,
                      factorize_5x5):
        InceptionBlockPartial = functools.partial(InceptionBlock, stride=1, norm=norm, act=act, bias=bias,
                                                  factorize_5x5=factorize_5x5, pool=nn.MaxPool2d)
        convs = [
            nn.Conv2d(in_channels, base_channels * 8, kernel_size=kernel_size_first, stride=stride_first,
                      padding=int((kernel_size_first - 1) / 2), bias=bias),
            norm(base_channels * 8),
            act(),
        ]
        if maxpool_first:
            convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        convs += [
            nn.Conv2d(base_channels * 8, base_channels * 24, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlockPartial(in_channels=base_channels * 24, out_channels_1x1=base_channels * 8,
                                  reduction_3x3=base_channels * 12, out_channels_3x3=base_channels * 16,
                                  reduction_5x5=base_channels * 2, out_channels_5x5=base_channels * 4,
                                  reduction_pool=base_channels * 4),
            InceptionBlockPartial(in_channels=base_channels * 32, out_channels_1x1=base_channels * 16,
                                  reduction_3x3=base_channels * 16, out_channels_3x3=base_channels * 24,
                                  reduction_5x5=base_channels * 4, out_channels_5x5=base_channels * 12,
                                  reduction_pool=base_channels * 8),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlockPartial(in_channels=base_channels * 60, out_channels_1x1=base_channels * 24,
                                  reduction_3x3=base_channels * 12, out_channels_3x3=base_channels * 26,
                                  reduction_5x5=base_channels * 2, out_channels_5x5=base_channels * 6,
                                  reduction_pool=base_channels * 8),
            InceptionBlockPartial(in_channels=base_channels * 64, out_channels_1x1=base_channels * 20,
                                  reduction_3x3=base_channels * 14, out_channels_3x3=base_channels * 28,
                                  reduction_5x5=base_channels * 3, out_channels_5x5=base_channels * 8,
                                  reduction_pool=base_channels * 8),
            InceptionBlockPartial(in_channels=base_channels * 64, out_channels_1x1=base_channels * 16,
                                  reduction_3x3=base_channels * 16, out_channels_3x3=base_channels * 32,
                                  reduction_5x5=base_channels * 3, out_channels_5x5=base_channels * 8,
                                  reduction_pool=base_channels * 8),
            InceptionBlockPartial(in_channels=base_channels * 64, out_channels_1x1=base_channels * 14,
                                  reduction_3x3=base_channels * 18, out_channels_3x3=base_channels * 36,
                                  reduction_5x5=base_channels * 4, out_channels_5x5=base_channels * 8,
                                  reduction_pool=base_channels * 8),
            InceptionBlockPartial(in_channels=base_channels * 66, out_channels_1x1=base_channels * 32,
                                  reduction_3x3=base_channels * 20, out_channels_3x3=base_channels * 40,
                                  reduction_5x5=base_channels * 4, out_channels_5x5=base_channels * 16,
                                  reduction_pool=base_channels * 16),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlockPartial(in_channels=base_channels * 104, out_channels_1x1=base_channels * 32,
                                  reduction_3x3=base_channels * 20, out_channels_3x3=base_channels * 40,
                                  reduction_5x5=base_channels * 4, out_channels_5x5=base_channels * 16,
                                  reduction_pool=base_channels * 16),
            InceptionBlockPartial(in_channels=base_channels * 104, out_channels_1x1=base_channels * 48,
                                  reduction_3x3=base_channels * 24, out_channels_3x3=base_channels * 48,
                                  reduction_5x5=base_channels * 6, out_channels_5x5=base_channels * 16,
                                  reduction_pool=base_channels * 16),
        ]

        return nn.Sequential(*convs), base_channels * 128

    @staticmethod
    def make_network(configs):
        norm = utils.get_norm(configs["norm"])
        act = utils.get_activation(configs["act"])

        default_params = {
            "in_channels": 3,
            "base_channels": 8,
            "kernel_size_first": 3,
            "stride_first": 1,
            "maxpool_first": False,
            "norm": norm,
            "act": act,
            "bias": False,
            "dropout": 0.4,
            "factorize_5x5": False,
            "num_classes": 10,
        }

        for key in default_params.keys():
            if key not in ["block", "norm", "act"] and key in configs:
                default_params[key] = configs[key]

        return Inception(**default_params)


@register.name_to_model.register("InceptionBN")
class InceptionBN(Inception):

    def __init__(self, in_channels, base_channels, kernel_size_first, stride_first, maxpool_first, norm, act, bias,
                 dropout, factorize_5x5, num_classes):
        super(InceptionBN, self).__init__(in_channels, base_channels, kernel_size_first, stride_first, maxpool_first,
                                          norm, act, bias, dropout, factorize_5x5, num_classes)

    @staticmethod
    def make_backbone(in_channels, base_channels, kernel_size_first, stride_first, maxpool_first, norm, act, bias,
                      factorize_5x5):
        InceptionBlockPartial = functools.partial(InceptionBlock, norm=norm, act=act, bias=bias,
                                                  factorize_5x5=factorize_5x5)
        convs = [
            nn.Conv2d(in_channels, base_channels * 8, kernel_size=kernel_size_first, stride=stride_first,
                      padding=int((kernel_size_first - 1) / 2), bias=bias),
            norm(base_channels * 8),
            act(),
        ]
        if maxpool_first:
            convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        convs += [
            nn.Conv2d(base_channels * 8, base_channels * 24, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlockPartial(in_channels=base_channels * 24, out_channels_1x1=base_channels * 8,
                                  reduction_3x3=base_channels * 8, out_channels_3x3=base_channels * 8,
                                  reduction_5x5=base_channels * 8, out_channels_5x5=base_channels * 12,
                                  reduction_pool=base_channels * 4, stride=1, pool=nn.AvgPool2d),
            InceptionBlockPartial(in_channels=base_channels * 32, out_channels_1x1=base_channels * 8,
                                  reduction_3x3=base_channels * 8, out_channels_3x3=base_channels * 12,
                                  reduction_5x5=base_channels * 8, out_channels_5x5=base_channels * 12,
                                  reduction_pool=base_channels * 8, stride=1, pool=nn.AvgPool2d),

            InceptionBlockPartial(in_channels=base_channels * 40, out_channels_1x1=base_channels * 0,
                                  reduction_3x3=base_channels * 16, out_channels_3x3=base_channels * 20,
                                  reduction_5x5=base_channels * 8, out_channels_5x5=base_channels * 12,
                                  reduction_pool=base_channels * 0, stride=2, pool=nn.MaxPool2d),
            InceptionBlockPartial(in_channels=base_channels * 72, out_channels_1x1=base_channels * 28,
                                  reduction_3x3=base_channels * 8, out_channels_3x3=base_channels * 12,
                                  reduction_5x5=base_channels * 12, out_channels_5x5=base_channels * 16,
                                  reduction_pool=base_channels * 16, stride=1, pool=nn.AvgPool2d),
            InceptionBlockPartial(in_channels=base_channels * 72, out_channels_1x1=base_channels * 24,
                                  reduction_3x3=base_channels * 12, out_channels_3x3=base_channels * 16,
                                  reduction_5x5=base_channels * 12, out_channels_5x5=base_channels * 16,
                                  reduction_pool=base_channels * 16, stride=1, pool=nn.AvgPool2d),
            InceptionBlockPartial(in_channels=base_channels * 72, out_channels_1x1=base_channels * 20,
                                  reduction_3x3=base_channels * 16, out_channels_3x3=base_channels * 20,
                                  reduction_5x5=base_channels * 16, out_channels_5x5=base_channels * 20,
                                  reduction_pool=base_channels * 16, stride=1, pool=nn.AvgPool2d),
            InceptionBlockPartial(in_channels=base_channels * 72, out_channels_1x1=base_channels * 12,
                                  reduction_3x3=base_channels * 16, out_channels_3x3=base_channels * 24,
                                  reduction_5x5=base_channels * 20, out_channels_5x5=base_channels * 24,
                                  reduction_pool=base_channels * 16, stride=1, pool=nn.AvgPool2d),

            InceptionBlockPartial(in_channels=base_channels * 128, out_channels_1x1=base_channels * 0,
                                  reduction_3x3=base_channels * 16, out_channels_3x3=base_channels * 24,
                                  reduction_5x5=base_channels * 24, out_channels_5x5=base_channels * 32,
                                  reduction_pool=base_channels * 0, stride=2, pool=nn.MaxPool2d),
            InceptionBlockPartial(in_channels=base_channels * 128, out_channels_1x1=base_channels * 44,
                                  reduction_3x3=base_channels * 24, out_channels_3x3=base_channels * 40,
                                  reduction_5x5=base_channels * 20, out_channels_5x5=base_channels * 28,
                                  reduction_pool=base_channels * 16, stride=1, pool=nn.AvgPool2d),
            InceptionBlockPartial(in_channels=base_channels * 128, out_channels_1x1=base_channels * 44,
                                  reduction_3x3=base_channels * 24, out_channels_3x3=base_channels * 40,
                                  reduction_5x5=base_channels * 24, out_channels_5x5=base_channels * 28,
                                  reduction_pool=base_channels * 16, stride=1, pool=nn.MaxPool2d),
        ]

        return nn.Sequential(*convs), base_channels * 128

    @staticmethod
    def make_network(configs):
        norm = utils.get_norm(configs["norm"])
        act = utils.get_activation(configs["act"])

        default_params = {
            "in_channels": 3,
            "base_channels": 8,
            "kernel_size_first": 3,
            "stride_first": 1,
            "maxpool_first": False,
            "norm": norm,
            "act": act,
            "bias": False,
            "dropout": 0.4,
            "factorize_5x5": False,
            "num_classes": 10,
        }

        for key in default_params.keys():
            if key not in ["block", "norm", "act"] and key in configs:
                default_params[key] = configs[key]

        return InceptionBN(**default_params)
