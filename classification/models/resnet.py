import torch.nn as nn
import torch.nn.functional as F

import register
from classification import utils
from classification import nnblock


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


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResBlock(nn.Module):
    """
    Implementation of ResBlock for supporting most
    settings of the module.
    """

    expansion = 1

    def __init__(self, in_channels, channel_span, kernel_size, stride, norm, act, down_sample, bias, use_short_cut,
                 pre_act, dropout, conv=nnblock.Conv2d):
        """
        Initialize ResBlock.

        Parameters
        ----------
        in_channels: int
          Number of channels of input tensor.
        channel_span: int
          Channel span of input tensor.
          This is deisgned for unified API of ResBlock and Bottleneck.
          For ResBlock, set channel_span=1 in general.
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
            3. "zero_padding": zero padding to match the channel size.
        bias: bool
          Whether to use bias in conv layer.
        use_short_cut: bool
          Whether to use shortcut connection in the block.
          Set this to False make the block as a simply two-layers conv block.
        pre_act: bool
          Whether to use pre-activation in the ResBlock.
        dropout: float
          Dropout rate.
        """
        super(ResBlock, self).__init__()

        # Using this padding value for kernel with size > 1
        # makes the output channel size irrelevant to the stride.
        p = int((kernel_size - 1) / 2)
        out_channels = int(in_channels * stride * channel_span)
        self.stride = stride
        self.use_short_cut = use_short_cut
        if not pre_act:
            self.convs = nn.Sequential(
                conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=p, bias=bias),
                norm(out_channels),
                act(),
                nn.Dropout(dropout),
                conv(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=p, bias=bias),
                norm(out_channels)
            )
        else:
            self.convs = nn.Sequential(
                norm(in_channels),
                act(),
                nn.Dropout(dropout),
                conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=p, bias=bias),
                norm(out_channels),
                act(),
                nn.Dropout(dropout),
                conv(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=p, bias=bias),
            )

        if use_short_cut:
            self.shortcut = None
            # shortcut tensor size will be mismatched with the main stream tensor
            if stride != 1:
                if down_sample == "conv":
                    if not pre_act:
                        self.shortcut = nn.Sequential(
                            conv(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias),
                            norm(out_channels)
                        )
                    else:
                        self.shortcut = nn.Sequential(
                            norm(in_channels),
                            conv(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias),
                        )
                elif down_sample == "interpolate":
                    pass  # implemented in "forward" method and do nothing here
                elif down_sample == "zero_padding":
                    self.shortcut = LambdaLayer(lambda x:
                                                F.pad(x[:, :, ::2, ::2],
                                                      (0, 0, 0, 0, out_channels // 4, out_channels // 4),
                                                      "constant", 0))
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
                x = x.repeat(1, self.stride, 1, 1)
                x1 += x

        return x1


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channel_span, kernel_size, stride, norm, act, down_sample, bias, use_short_cut,
                 pre_act, dropout, conv=nnblock.Conv2d):
        """
        Initialize Bottleneck.

        Parameters
        ----------
        in_channels: int
          Number of channels of input tensor.
        channel_span: int
          Channel span of input tensor.
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
            3. "zero_padding": zero padding to match the channel size.
        bias: bool
          Whether to use bias in conv layer.
        use_short_cut: bool
          Whether to use shortcut connection in the block.
          Set this to False make the block as a simply two-layers conv block.
        pre_act: bool
          Whether to use pre-activation in the ResBlock.
        dropout: float
          Dropout rate.
        """
        super().__init__()
        self.stride = stride
        self.use_short_cut = use_short_cut

        # Using this padding value for kernel with size > 1
        # makes the output channel size irrelevant to the stride.
        p = int((kernel_size - 1) / 2)

        hidden_channels = int(in_channels * channel_span)
        out_channels = int(hidden_channels * Bottleneck.expansion)

        if not pre_act:
            self.convs = nn.Sequential(
                conv(in_channels, hidden_channels, kernel_size=1, bias=bias),
                norm(hidden_channels),
                act(),
                nn.Dropout(dropout),
                conv(hidden_channels, hidden_channels, stride=stride, kernel_size=kernel_size, padding=p,
                     bias=bias),
                norm(hidden_channels),
                act(),
                nn.Dropout(dropout),
                conv(hidden_channels, out_channels, kernel_size=1, bias=bias),
                norm(out_channels),
            )
        else:
            self.convs = nn.Sequential(
                norm(in_channels),
                act(),
                nn.Dropout(dropout),
                conv(in_channels, hidden_channels, kernel_size=1, bias=bias),
                norm(hidden_channels),
                act(),
                nn.Dropout(dropout),
                conv(hidden_channels, hidden_channels, stride=stride, kernel_size=kernel_size, padding=p,
                     bias=bias),
                norm(hidden_channels),
                act(),
                nn.Dropout(dropout),
                conv(hidden_channels, out_channels, kernel_size=1, bias=bias),
            )

        if use_short_cut:
            self.shortcut = None
            # shortcut tensor size will be mismatched with the main stream tensor
            if in_channels != out_channels:
                if down_sample == "conv":
                    if not pre_act:
                        self.shortcut = nn.Sequential(
                            conv(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias),
                            norm(out_channels)
                        )
                    else:
                        self.shortcut = nn.Sequential(
                            norm(in_channels),
                            conv(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias),
                        )
                elif down_sample == "interpolate":
                    pass  # implemented in "forward" method and do nothing here
                elif down_sample == "zero_padding":
                    self.shortcut = LambdaLayer(lambda x:
                                                F.pad(x[:, :, ::2, ::2],
                                                      (0, 0, 0, 0, out_channels // 4, out_channels // 4),
                                                      "constant", 0))
                else:
                    raise NotImplementedError
            else:
                self.shortcut = Identity()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.in_channels = in_channels

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
                x = x.repeat(1, int(self.out_channels / self.in_channels), 1, 1)
                x1 += x

        return x1


@register.name_to_model.register("ResNet")
class ResNet(nn.Module):
    """
    Implementation of the ResNet paper:
      1. "Deep Residual Learning for Image Recognition"
      2. "Identity Mappings in Deep Residual Networks"
    """

    def __init__(self, block, n_blocks_list, stride_list, in_channels, hidden_channels, kernel_size,
                 kernel_size_first, stride_first, use_bn_first, use_act_first, norm, act, down_sample,
                 bias, use_short_cut, use_maxpool, num_classes, pre_act, dropout, use_out_act, conv=nnblock.Conv2d):
        """
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
        use_bn_first: bool
          Whether to use BN in the first layer.
        use_act_first: bool
          Whether to use activation function in the first layer.
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
            3. "zero_padding": zero padding to match the channel size.
        bias: bool
          Whether to use bias in conv layer.
        use_short_cut: bool
          Whether to use shortcut connection in the block.
          Set this to False make the block as a simply two-layers conv block.
        use_maxpool: bool
          Whether to use maxpool in the network.
        num_classes: int
          Number of classes.
        pre_act: bool
          Whether to use pre-activation in the ResBlock.
        dropout: float
          Dropout rate.
        use_out_act: bool
          Whether to use activation on the output of ResBlock.
        """

        super().__init__()
        self.convs, out_channels = ResNet.make_backbone(block, n_blocks_list, stride_list, in_channels,
                                                        hidden_channels,
                                                        kernel_size, kernel_size_first, stride_first, use_bn_first,
                                                        use_act_first, norm, act, down_sample, bias, use_short_cut,
                                                        use_maxpool,
                                                        pre_act, dropout, use_out_act, conv)
        self.act_last = None if not pre_act else act()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels * block.expansion, num_classes)

    def forward(self, x):
        output = self.convs(x) if self.act_last is None else self.act_last(self.convs(x))
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

    @staticmethod
    def _make_res_part(block, in_channels, channel_span, kernel_size, stride, norm, act, down_sample, bias,
                       use_short_cut, n_blocks, pre_act, dropout, use_out_act, conv=nnblock.Conv2d):
        """
        Utility function for constructing res part in resnet.

        Parameters
        ----------
        block: nn.Module
          ResBlock or Bottleneck used in network.
        in_channels: int
          Number of channels of input tensor.
        channel_span: int
          Channel span of input tensor.
          This is deisgned for unified API of ResBlock and Bottleneck.
          For ResBlock, set channel_span=1 in general.
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
            3. "zero_padding": zero padding to match the channel size.
        bias: bool
          Whether to used bias in conv layer.
        use_short_cut: bool
          Whether to used shortcut connection in the block.
          Set this to False make the block as a simply two-layers conv block.
        n_blocks: int
          number of blocks in the part.
        pre_act: bool
          Whether to use pre-activation in the ResBlock.
        dropout: float
          Dropout rate.
        use_out_act: bool
          Whether to use activation on the output of ResBlock.
        """

        # The first res block is with the specified "stride", and
        # all others are with stride=1.
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            if i == 0:
                layers.append(
                    block(in_channels, channel_span, kernel_size, stride, norm, act, down_sample, bias, use_short_cut,
                          pre_act, dropout, conv))
                if block.expansion == 1:
                    in_channels = int(in_channels * stride * block.expansion)
                else:
                    in_channels = int(in_channels * block.expansion * channel_span)
            else:
                layers.append(
                    block(in_channels, 1 / float(block.expansion), kernel_size, stride, norm, act, down_sample, bias,
                          use_short_cut, pre_act, dropout, conv))
            if not pre_act and use_out_act:
                layers.append(act())

        return layers

    @staticmethod
    def make_backbone(block, n_blocks_list, stride_list, in_channels, hidden_channels, kernel_size,
                      kernel_size_first, stride_first, use_bn_first, use_act_first, norm, act, down_sample,
                      bias, use_short_cut, use_maxpool, pre_act, dropout, use_out_act, conv=nnblock.Conv2d):
        """
        Construct resnet-like backbone.

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
        use_bn_first: bool
          Whether to use BN in the first layer.
        use_act_first: bool
          Whether to use activation function in the first layer.
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
            3. "zero_padding": zero padding to match the channel size.
        bias: bool
          Whether to use bias in conv layer.
        use_short_cut: bool
          Whether to use shortcut connection in the block.
          Set this to False make the block as a simply two-layers conv block.
        use_maxpool: bool
          Whether to use maxpool in the network.
        pre_act: bool
          Whether to use pre-activation in the ResBlock.
        dropout: float
          Dropout rate.
        use_out_act: bool
          Whether to use activation on the output of ResBlock.

        Return
        ----------
        (convs, hidden_channels): Tuple
          convs: nn.Module
            The constructed backbone.
          hidden_channels:
            Channels of the output of the backbone.
        """
        convs = [
            conv(in_channels, hidden_channels, kernel_size=kernel_size_first, stride=stride_first,
                 padding=int((kernel_size_first - 1) / 2), bias=bias),
        ]
        if use_bn_first:
            convs.append(nn.BatchNorm2d(hidden_channels))
        if use_act_first:
            convs.append(act())

        if use_maxpool:
            convs += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        expansion = block.expansion

        for i, (n_blocks, stride) in enumerate(zip(n_blocks_list, stride_list)):
            # For the first part, stride=1 and channel_span=1
            # I have no simple idea to make the code pythonic
            if i == 0:
                convs += ResNet._make_res_part(block, hidden_channels, 1, kernel_size, stride, norm, act,
                                               down_sample, bias, use_short_cut, n_blocks, pre_act, dropout,
                                               use_out_act, conv)
                if block.expansion == 1:
                    hidden_channels = hidden_channels * stride
            else:
                if block.expansion == 4:
                    convs += ResNet._make_res_part(block, hidden_channels * 4, 0.5, kernel_size,
                                                   stride, norm, act, down_sample, bias, use_short_cut, n_blocks,
                                                   pre_act, dropout, use_out_act, conv)
                    hidden_channels = hidden_channels * 2
                else:
                    convs += ResNet._make_res_part(block, hidden_channels, 1, kernel_size, stride, norm, act,
                                                   down_sample,
                                                   bias, use_short_cut, n_blocks, pre_act, dropout, use_out_act, conv)
                    hidden_channels = hidden_channels * stride

        convs = nn.Sequential(*convs)
        return convs, hidden_channels

    @staticmethod
    def make_network(configs):
        conv = nnblock.get_conv(configs)
        norm = utils.get_norm(configs["norm"])
        act = utils.get_activation(configs["act"])

        if configs["block"] == "ResBlock":
            block = ResBlock
        elif configs["block"] == "Bottleneck":
            block = Bottleneck
        else:
            raise NotImplementedError

        default_params = {
            "conv": conv,
            "block": block,
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
            "dropout": 0,
            "use_out_act": True,
        }

        for key in default_params.keys():
            if key not in ["conv", "block", "norm", "act"] and key in configs:
                default_params[key] = configs[key]

        return ResNet(**default_params)
