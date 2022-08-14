import torch.nn as nn
from classification.models import acb


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

    @staticmethod
    def get_conv(configs):
        return Conv2d


def get_conv(configs):
    conv_name = configs["conv"] if "conv" in configs else "Conv2d"
    assert conv_name in convs.keys()
    conv = convs[conv_name]
    conv = conv.get_conv(configs)
    return conv


convs = {
    "Conv2d": Conv2d,
    "ACBlock": acb.ACBlock,
}
