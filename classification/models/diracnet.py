from torch import nn
from collections import OrderedDict

import register
from classification import utils
from classification.models.vgg import VGG
from classification.models.plainnet import PlainNet
from nn_module.conv.convs import Conv2d


@register.name_to_model.register("DiracNet")
class DiracNet(nn.Module):

    def __init__(self, in_channels, hidden_channels, n_blocks_list, stride_list, stride_factor, num_classes,
                 last_act, pool_size, conv=Conv2d, out_feats=None):
        super().__init__()
        assert len(n_blocks_list) > 0
        assert len(n_blocks_list) == len(stride_list)
        assert isinstance(stride_factor, int) or len(stride_factor) == len(n_blocks_list)

        if isinstance(stride_factor, int):
            stride_factor = [stride_factor] * len(n_blocks_list)
            stride_factor = [PlainNet._get_stride_factor(stride, sf) for stride, sf in zip(stride_list, stride_factor)]

        self.backbone = [conv(in_channels, hidden_channels, stride=1)]

        # Define the first to last (exclude) groups
        vgg_backbone, hidden_channels = VGG.make_backbone(n_blocks_list[:-1], stride_list[:-1], hidden_channels,
                                                          hidden_channels, stride_factor[:-1], pool_size, conv)
        self.backbone += vgg_backbone

        sf = PlainNet._get_stride_factor(stride_list[-1], stride_factor[-1])
        out_channels = int(hidden_channels * sf)
        last_group = PlainNet.make_plain_part(hidden_channels, out_channels, 1, n_blocks_list[-1], conv)
        self.backbone += last_group
        self.backbone = nn.Sequential(*self.backbone)
        self.last_act = last_act()
        self.out_feats = out_feats

        self.num_classes = num_classes
        if num_classes == 2:
            num_classes = 1
        if num_classes > 0:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        if self.out_feats is None or self.out_feats == "None":
            output = self.last_act(self.backbone(x))
            if self.num_classes is not None and self.num_classes > 0:
                output = self.avg_pool(output)
                output = output.view(output.size(0), -1)
                output = self.fc(output)
            return output
        else:
            outputs = OrderedDict()
            for i, (name, layer) in enumerate(self.backbone.named_children()):
                x = layer(x)
                if i in self.out_feats:
                    outputs[f"layer_{i}"] = x

            output = self.last_act(x)
            if self.num_classes is not None and self.num_classes > 0:
                output = self.avg_pool(output)
                output = output.view(output.size(0), -1)
                output = self.fc(output)
            outputs["layer_last"] = output
            return outputs

    @staticmethod
    def make_network(configs):
        conv = register.get_conv(configs, "conv")
        act = register.get_activation(configs["last_act"])

        default_params = {
            "in_channels"    : 3,
            "hidden_channels": 64,
            "n_blocks_list"  : [2, 2, 2, 2],
            "stride_list"    : [2, 1, 1, 1],
            "stride_factor"  : 2,
            "num_classes"    : 10,
            "last_act"       : act,
            "pool_size"      : 3,
            "conv"           : conv,
        }
        default_params = utils.set_params(default_params, configs, excluded_keys=["conv", "last_act"])
        return DiracNet(**default_params)