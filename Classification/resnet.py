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

    def __init__(self, in_channels, channel_span, kernel_size, stride, norm, act, down_sample, bias, use_short_cut):
        """
        Initialize ResBlock.

        Parameters
        ----------
        in_channels: int
            Number of channels of input tensor.
        channel_span: int
            Number of channels of hidden tensor.
            This is deisgned for unified API of ResBlock and Bottleneck.
        kernel_size: int
            Kernel size of convolution operator.
        stride: int
            Stride of the first conv layer in the block.
            The output channel of the first conv layer is: in_channels*stride
        norm: nn.Module
            The normalization method used in the block.
            Set this to "IdentityNorm" to disable normalization.
        act: nn.Module
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
        out_channels = int(in_channels * stride * channel_span)
        self.stride = stride
        self.use_short_cut = use_short_cut
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=p, bias=bias)
        self.norm1 = norm(out_channels)
        self.act1 = act()
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
        x1 = self.act1(x1)
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

    def __init__(self, in_channels, channel_span, kernel_size, stride, norm, act, down_sample, bias, use_short_cut):
        """
        Initialize Bottleneck.

        Parameters
        ----------
        in_channels: int
            Number of channels of input tensor.
        channel_span: int
            Number of channels of hidden tensor.
        kernel_size: int
            Kernel size of convolution operator.
        stride: int
            Stride of the first conv layer in the block.
            The output channel of the first conv layer is: in_channels*stride
        norm: nn.Module
            The normalization method used in the block.
            Set this to "IdentityNorm" to disable normalization.
        act: nn.Module
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
        self.stride = stride
        self.use_short_cut = use_short_cut

        # Using this padding value for kernel with size > 1
        # makes the output channel size irrelevant to the stride.
        p = int((kernel_size - 1) / 2) 

        hidden_channels = int(in_channels * channel_span)
        out_channels = int(hidden_channels * Bottleneck.expansion)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=bias),
            norm(hidden_channels),
            act(),
            nn.Conv2d(hidden_channels, hidden_channels, stride=stride, kernel_size=kernel_size, padding=p, bias=bias),
            norm(hidden_channels),
            act(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=bias),
            norm(out_channels),
        )

        if use_short_cut:
            self.shortcut = None
            # shortcut tensor size will be mismatched with the main stream tensor
            if in_channels != out_channels:
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
        
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

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
                x = x.repeat(1, int(self.out_channels / self.hidden_channels), 1, 1)
                x1 += x

        return x1


class ResNet(nn.Module):
    """
    Implementation of the ResNet paper:
    "Deep Residual Learning for Image Recognition"
    """

    def __init__(self, block, n_blocks_list, stride_list, in_channels, hidden_channels, kernel_size, 
                 kernel_size_first, stride_first, norm, act, down_sample, bias, use_short_cut, use_maxpool, 
                 num_classes):
        """
        Initialize Bottleneck.

        Parameters
        ----------
        block: nn.Module
            The building block. Possible values are: ResBlock and Bottleneck. 
        n_blocks_list: List(int)
            The list (arbitrary length) specifies number of building blocks in each part.
        stride_list: List(int)
            The list (arbitrary length) specifies stride of building blocks in each part.
            Ensure that len(n_blocks_list) == len(n_blocks_list).
        in_channels: int
            Number of channels of input tensor.
        hidden_channels: int
            Number of channels of hidden tensor.
        kernel_size: int
            Kernel size of convolution operator in building block.
        kernel_size_first: int
            Kernel size of the first convolution operator.
            The typical value is kernel_size_first=7.
        stride_first: int
            Stride of the first conv layer in the block.
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
        use_maxpool:
            Whether to used maxpool in the network.
        num_classes:
            Number of classes.
        """

        super().__init__()

        self.convs = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size_first, stride=stride_first, padding=int((kernel_size_first-1)/2), bias=False),
            nn.BatchNorm2d(hidden_channels),
            act(),
        ]
        if use_maxpool:
            self.convs += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        expansion = block.expansion

        for i, (n_blocks, stride) in enumerate(zip(n_blocks_list, stride_list)):
            # For the first part, stride=1 and channel_span=1
            # I have no simple idea to make the code pythonic
            if i == 0:
                self.convs += ResNet._make_res_part(block, hidden_channels, 1, kernel_size, stride, norm, act,
                                                    down_sample, bias, use_short_cut, n_blocks)
                if block.expansion == 1:
                    hidden_channels = hidden_channels * stride
            else:          
                if block.expansion == 4:
                    self.convs += ResNet._make_res_part(block, hidden_channels * 4, 0.5, kernel_size,
                                                        stride, norm, act, down_sample, bias, use_short_cut, n_blocks)
                    hidden_channels = hidden_channels * 2
                else:
                    self.convs += ResNet._make_res_part(block, hidden_channels, 1, kernel_size, stride, norm, act, down_sample,
                                                        bias, use_short_cut, n_blocks)              
                    hidden_channels = hidden_channels * stride      

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
    def _make_res_part(block, in_channels, channel_span, kernel_size, stride, norm, act, down_sample, bias,
                       use_short_cut, n_blocks):
        """
        Utility function for constructing res part in resnet.

        Parameters
        ----------
        block: nn.Module
            ResBlock or Bottleneck used in network.
        in_channels: int
            Number of channels of input tensor.
        channel_span: int
            Number of channels of hidden tensor.
        kernel_size: int
            Kernel size of convolution operator.
        stride: int
            Stride of the first conv layer in the block.
            The output channel of the first conv layer is: in_channels*stride
        norm: nn.Module
            The normalization method used in the block.
            Set this to "IdentityNorm" to disable normalization.
        act: nn.Module
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
                layers.append(block(in_channels, channel_span, kernel_size, stride, norm, act, down_sample, bias, use_short_cut))
                if block.expansion == 1:
                    in_channels = int(in_channels * stride * block.expansion)
                else:
                    in_channels = int(in_channels * block.expansion * channel_span)
            else:
                layers.append(block(in_channels, 1 / float(block.expansion), kernel_size, stride, norm, act, down_sample, bias, use_short_cut))

            layers.append(act())

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
            "stride_list": [2, 1, 1, 1],
            "in_channels": 3,
            "hidden_channels": 64,
            "kernel_size": 3,
            "kernel_size_first": 3,
            "stride_first": 1,
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
