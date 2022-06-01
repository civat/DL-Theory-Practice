import torch.nn as nn
import torch.nn.functional as F

import utils


class Identity(nn.Module):
    """
    Identity mapping.
    This is generally used as "Identity activation" in networks for
    convenient implementation of "no activation".
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class IdentityNorm(nn.Module):
    """
        Identity norm.
        This is generally used as "Identity norm" in networks for
        convenient implementation of "no normalization".
        """

    def __init__(self, in_channels):
        super(IdentityNorm, self).__init__()

    def forward(self, x):
        return x


class ResBlock(nn.Module):
    """
    Complex implementation of ResBlock for supporting most
    settings of the module.
    """

    expansion = 1

    def __init__(self, in_channels, kernel_size, stride, norm, act, down_sample, bias, use_short_cut):
        """
        Initialize ResBlock.

        Parameters
        ----------
        in_channels: int
            Number of channels of input tensor.
        kernel_size: int
            Kernel size of convolution operator.
        stride: int
            Stride of the first conv layer in the block.
            The output channel of the first conv layer is: in_channels*stride
        norm: nn.Module
            The normalization method used in the block.
            Set this to "IdentityNorm" to disable normalization.
        act: nn.Module object
            The activation function used in the block.
        down_sample: string
            The down-sampling method used in the block when
            the shortcut tensor size is mismatched with the
            main stream tensor.
            The available values are:
                1. "conv": use 1x1 conv layer to match the size;
                2. "interpolate": use bi-linear interpolation to match the
                   width and height of the feature maps. Copy the resized feature
                   map several times to match the channel size.
        bias: bool
            Whether to used bias in conv layer.
        use_short_cut: bool
            Whether to used shortcut connection in the block.
            Set this to False make the block as a simply two-layers conv block.
        """
        super(ResBlock, self).__init__()

        # Using this padding value for kernel with size > 1
        # makes the output channel size irrelevant to the stride.
        p = int((kernel_size - 1) / 2)
        out_channels = in_channels * stride
        self.act = act
        self.stride = stride
        self.use_short_cut = use_short_cut
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=p, bias=bias)
        self.norm1 = norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=p, bias=bias)
        self.norm2 = norm(out_channels)

        if use_short_cut:
            self.shortcut = None
            # shortcut tensor size will be mismatched with the main stream tensor
            if stride != 1:
                if down_sample == "conv":
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias),
                        norm(out_channels)
                    )
                elif down_sample == "interpolate":
                    pass  # implemented in "forward" method and do nothing here
                else:
                    raise NotImplementedError
            else:
                self.shortcut = Identity()

    def forward(self, x):
        x1 = self.norm1(self.conv1(x))
        x1 = self.act(x1)
        x1 = self.norm2(self.conv2(x1))

        if self.use_short_cut:
            if self.shortcut is not None:
                x1 += self.shortcut(x)
            else:
                # The only reason that we get here is the down_sample=="interpolate".
                # So we implement the method here.
                h, w = x1.size(-2), x1.size(-1)
                x = F.interpolate(x, size=(h, w), mode="bilinear")
                x = x.repeat(1, self.stride, 1, 1)
                x1 += x

        return x1


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, kernel_size, stride, norm, act, down_sample, bias, use_short_cut):
        """
        Initialize Bottleneck.

        Parameters
        ----------
        in_channels: int
            Number of channels of input tensor.
        kernel_size: int
            Kernel size of convolution operator.
        stride: int
            Stride of the first conv layer in the block.
            The output channel of the first conv layer is: in_channels*stride
        norm: nn.Module
            The normalization method used in the block.
            Set this to "IdentityNorm" to disable normalization.
        act: nn.Module object
            The activation function used in the block.
        down_sample: string
            The down-sampling method used in the block when
            the shortcut tensor size is mismatched with the
            main stream tensor.
            The available values are:
                1. "conv": use 1x1 conv layer to match the size;
                2. "interpolate": use bi-linear interpolation to match the
                   width and height of the feature maps. Copy the resized feature
                   map several times to match the channel size.
        bias: bool
            Whether to used bias in conv layer.
        use_short_cut: bool
            Whether to used shortcut connection in the block.
            Set this to False make the block as a simply two-layers conv block.
        """
        super().__init__()
        self.act = act
        self.stride = stride

        # Using this padding value for kernel with size > 1
        # makes the output channel size irrelevant to the stride.
        p = int((kernel_size - 1) / 2)
        if stride == 1:
            hidden_channels = in_channels
        else:
            hidden_channels = int(in_channels / 2)
        out_channels = hidden_channels * Bottleneck.expansion
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=bias),
            norm(hidden_channels),
            act,
            nn.Conv2d(hidden_channels, hidden_channels, stride=stride, kernel_size=kernel_size, padding=p, bias=bias),
            norm(hidden_channels),
            act,
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=bias),
            norm(out_channels),
        )

        if use_short_cut:
            self.shortcut = None
            # shortcut tensor size will be mismatched with the main stream tensor
            if stride != 1:
                if down_sample == "conv":
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias),
                        norm(out_channels)
                    )
                elif down_sample == "interpolate":
                    pass  # implemented in "forward" method and do nothing here
                else:
                    raise NotImplementedError
            else:
                self.shortcut = Identity()

    def forward(self, x):
        x1 = self.convs(x)

        if self.use_short_cut:
            if self.shortcut is not None:
                x1 += self.shortcut(x)
            else:
                # The only reason that we get here is the down_sample=="interpolate".
                # So we implement the method here.
                h, w = x1.size(-2), x1.size(-1)
                x = F.interpolate(x, size=(h, w), mode="bilinear")
                x = x.repeat(1, self.stride * Bottleneck.expansion, 1, 1)
                x1 += x

        return x1


class ResNet(nn.Module):

    def __init__(self, block, n_blocks_list, in_channels, hidden_channels, kernel_size, stride,
                 norm, act, down_sample, bias, use_short_cut, use_maxpool, num_classes):
        super().__init__()

        self.convs = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm2d(hidden_channels),
            act,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
        if use_maxpool:
            self.convs += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        expansion = block.expansion

        for i, n_blocks in enumerate(n_blocks_list):
            # I have no simple idea to make the code pythonic

            if i == 0:
                self.convs += ResNet._make_res_part(block, hidden_channels, hidden_channels, kernel_size, 1, norm, act,
                                                    down_sample, bias, use_short_cut, n_blocks)
            else:
                if block.expansion == 4:
                    self.convs += ResNet._make_res_part(block, hidden_channels * 4, hidden_channels * 2, kernel_size,
                                                        2, norm, act, down_sample, bias, use_short_cut, n_blocks)
                else:
                    self.convs += ResNet._make_res_part(block, hidden_channels, 0, kernel_size, 2, norm, act, down_sample,
                                                        bias, use_short_cut, n_blocks)
                hidden_channels = hidden_channels * 2

        self.convs = nn.Sequential(*self.convs)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_channels * expansion, num_classes)

    def forward(self, x):
        output = self.convs(x)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

    @staticmethod
    def _make_res_part(block, in_channels, kernel_size, stride, norm, act, down_sample, bias,
                       use_short_cut, n_blocks):
        """
        Utility function for constructing res part in resnet.

        Parameters
        ----------
        block: nn.Module
            ResBlock or Bottleneck used in network.
        in_channels: int
            Number of channels of input tensor.
        kernel_size: int
            Kernel size of convolution operator.
        stride: int
            Stride of the first conv layer in the block.
            The output channel of the first conv layer is: in_channels*stride
        norm: nn.Module
            The normalization method used in the block.
            Set this to "IdentityNorm" to disable normalization.
        act: nn.Module object
            The activation function used in the block.
        down_sample: string
            The down-sampling method used in the block when
            the shortcut tensor size is mismatched with the
            main stream tensor.
            The available values are:
                1. "conv": use 1x1 conv layer to match the size;
                2. "interpolate": use bi-linear interpolation to match the
                   width and height of the feature maps. Copy the resized feature
                   map several times to match the channel size.
        bias: bool
            Whether to used bias in conv layer.
        use_short_cut: bool
            Whether to used shortcut connection in the block.
            Set this to False make the block as a simply two-layers conv block.
        n_blocks: int
            number of blocks in the part.
        """

        # The first res block is with the specified "stride", and
        # all others are with stride=1.
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            if i == 0:
                layers.append(block(in_channels, kernel_size, stride, norm, act, down_sample, bias, use_short_cut))
                if block.expansion == 1:
                    in_channels = in_channels * stride * block.expansion
                else:
                    in_channels = in_channels * block.expansion
            else:
                layers.append(block(in_channels, kernel_size, stride, norm, act, down_sample, bias, use_short_cut))
            layers.append(act)

        return layers

    @staticmethod
    def make_network(configs):
        norm = utils.get_norm(configs["norm"])
        act = utils.get_activation(configs["act"])

        if configs["block"] == "ResBlock":
            block = ResBlock
        elif configs["block"] == "Bottleneck":
            block = Bottleneck
        else:
            raise NotImplementedError
        default_params = {
            "block": block,
        }

        default_params = {
            "block": block,
            "n_blocks_list": [2, 2, 2, 2],
            "in_channels": 3,
            "hidden_channels": 64,
            "kernel_size": 3,
            "stride": 2,
            "norm": norm,
            "act": act,
            "down_sample": "conv",
            "bias": False,
            "use_short_cut": True,
            "use_maxpool": True,
            "num_classes": 10
        }

        for key in default_params.keys():
            if key not in ["block", "norm", "act"] and key in configs:
                default_params[key] = configs[key]

        return ResNet(**default_params)
