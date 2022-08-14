import torch
import torch.nn as nn

from classification import utils
from classification.models import resnet


class WideBlock(nn.Module):

    def __init__(self, in_channels, kernel_size, stride, norm, act, down_sample, bias, 
                 use_short_cut, n_blocks, pre_act, width):
        super(WideBlock, self).__init__()
        self.blocks = []
        self.blocks = [nn.ModuleList(
            resnet.ResNet._make_res_part(resnet.ResBlock, in_channels, 1, kernel_size, stride, norm, act,
                                         down_sample, bias, use_short_cut, n_blocks, pre_act) for _ in range(width))]
        print(self.blocks)
        self.blocks = nn.ModuleList(self.blocks)

        if not act:
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_channels*stride*width, in_channels*stride, kernel_size=1, bias=bias),
                norm(in_channels*stride),
                act(),
            )
        else:
            self.conv1x1 = nn.Sequential(
                norm(in_channels*stride),
                act(),
                nn.Conv2d(in_channels*stride*width, in_channels*stride, kernel_size=1, bias=bias),
            )

    def forward(self, x):
        out = [block(x) for block in self.blocks]
        out = torch.concat(out, dim=1)
        out = self.conv1x1(out)
        return out


class WideNet(nn.Module):

    def __init__(self, n_blocks_list, stride_list, in_channels, hidden_channels, kernel_size, 
                 kernel_size_first, stride_first, use_bn_first, use_act_first, norm, act, down_sample, 
                 bias, use_short_cut, use_maxpool, num_classes, pre_act, width):
        super(WideNet, self).__init__()
        self.convs = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size_first, stride=stride_first, padding=int((kernel_size_first-1)/2), bias=False),
        ]
        if use_bn_first:
            self.convs.append(nn.BatchNorm2d(hidden_channels))
        if use_act_first:
            self.convs.append(act())
            
        if use_maxpool:
            self.convs += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        
        for n_blocks, stride in zip(n_blocks_list, stride_list):
            self.convs.append(WideBlock(hidden_channels, kernel_size, stride, norm, act, down_sample, bias, 
                                        use_short_cut, n_blocks, pre_act, width))
            hidden_channels *= stride
        
        self.convs = nn.Sequential(*self.convs)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x):
        output = self.convs(x)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

    @staticmethod
    def make_network(configs):
        norm = utils.get_norm(configs["norm"])
        act = utils.get_activation(configs["act"])

        default_params = {
            "n_blocks_list": [2, 2, 2, 2],
            "stride_list": [2, 1, 1, 1],
            "in_channels": 3,
            "hidden_channels": 64,
            "kernel_size": 3,
            "kernel_size_first": 3,
            "stride_first": 1,
            "use_bn_first": True,
            "use_act_first": True,
            "norm": norm,
            "act": act,
            "down_sample": "conv",
            "bias": False,
            "use_short_cut": True,
            "use_maxpool": True,
            "num_classes": 10,
            "pre_act": False,
            "width": 5,
        }

        for key in default_params.keys():
            if key not in ["block", "norm", "act"] and key in configs:
                default_params[key] = configs[key]

        return WideNet(**default_params)
