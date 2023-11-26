import torch.nn as nn
from collections import OrderedDict

import register
from classification import utils
from classification.models.plainnet import PlainNet
from nn_module.conv.convs import Conv2d


@register.name_to_model.register("GNetwork")
class GeneralNetwork(nn.Module):

    def __init__(self, in_channels, hidden_channels, n_blocks_list, stride_list, stride_factor_list,
                    num_classes, convs, fcs, last_act, out_feats=None):
        super().__init__()
        assert len(n_blocks_list) > 0
        assert len(n_blocks_list) == len(stride_list) == len(stride_factor_list) == len(convs)
        self.n = len(n_blocks_list)
        self.out_feats = out_feats

        self.convs, out_channels = self.make_backbone(n_blocks_list, stride_list, in_channels, hidden_channels,
                                                      stride_factor_list, convs)
        self.last_act = last_act

        # for binary classification task, we use BCE loss so only one output logit is needed.
        self.num_classes = num_classes
        if num_classes == 2:
            num_classes = 1
        if num_classes > 0:
            if fcs is None:
                raise Exception("Fully connected layers are not specified by the key \"fcs\"!")
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fcs = [conv() for conv in fcs]
            self.fcs = nn.Sequential(*self.fcs)

    @staticmethod
    def make_backbone(n_blocks_list, stride_list, in_channels, hidden_channels, stride_factor_list, conv_list):
        convs = []

        for i, (n_blocks, stride, stride_factor, conv) in enumerate(zip(n_blocks_list, stride_list, stride_factor_list, conv_list)):
            if i != 0:
                hidden_channels = int(in_channels * stride_factor)
            convs += PlainNet.make_plain_part(in_channels, hidden_channels, stride, n_blocks, conv)
            in_channels = hidden_channels

        convs = nn.Sequential(*convs)
        return convs, hidden_channels

    def forward(self, x):
        if self.out_feats is None or self.out_feats == "None":
            output = self.last_act(self.convs(x))
            if self.num_classes > 0:
                output = self.avg_pool(output)
                output = self.fcs(output)
                output = output.view(output.size(0), -1)
            return output
        else:
            outputs = OrderedDict()
            for i, (name, layer) in enumerate(self.convs.named_children()):
                x = layer(x)
                for i in self.out_feats:
                    outputs[f"layer_P{i}"] = x
            output = self.last_act(x)
            if self.num_classes > 0:
                output = self.avg_pool(output)
                output = self.fcs(output)
                output = output.view(output.size(0), -1)
            outputs["layer_last"] = output
            return outputs

    @staticmethod
    def make_network(configs):
        config_convs = configs["convs"]
        conv_list = []
        for i in range(len(config_convs)):
            if f"conv{i}" not in config_convs:
                raise Exception(f"The key \"conv{i}\" is not specified.")
            else:
                conv_list += register.get_conv(config_convs, f"conv{i}")

        fc_list = None
        if "fcs" in configs:
            config_fcs = configs["fcs"]
            fc_list = []
            for i in range(len(config_fcs)):
                if f"fc{i}" not in config_fcs:
                    raise Exception(f"The key \"fc{i}\" is not specified.")
                else:
                    fc_list += register.get_conv(config_fcs, f"fc{i}")

        act = register.get_activation(configs["last_act"])

        default_params = {
            "in_channels"       : 3,
            "hidden_channels"   : 64,
            "n_blocks_list"     : [2, 2, 2, 2],
            "stride_list"       : [2, 1, 1, 1],
            "stride_factor_list": [2, 2, 2, 2],
            "num_classes"       : 10,
            "convs"             : conv_list,
            "fcs"               : fc_list,
            "last_act"          : act,
            "out_feats"         : None
        }
        default_params = utils.set_params(default_params, configs, excluded_keys=["convs", "fcs", "last_act"])
        return GeneralNetwork(**default_params)