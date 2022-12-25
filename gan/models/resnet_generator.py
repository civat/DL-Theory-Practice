import torch.nn as nn

import register
from classification.models.resnet import Identity
from classification.utils import get_norm
from classification.utils import get_activation


class ResBlockTranspose(nn.Module):
    """Implementation of ResBlock with transposed convolution."""

    def __init__(self, in_channels, kernel_size, stride, norm, act, up_sample, bias, use_short_cut,
                 pre_act, dropout, padding_mode):
        """
        Initialize transposed ResBlock.

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
        act: nn.Module
          The activation function used in the block.
        up_sample: str
          The up-sampling method used in the block when
          the shortcut tensor size is mismatched with the
          main stream tensor.
          The available values are (only 1x1 conv is supported now):
            1. "conv": use 1x1 conv layer to match the size;
        bias: bool
          Whether to use bias in conv layer.
        use_short_cut: bool
          Whether to use shortcut connection in the block.
          Set this to False make the block as a simply two-layers conv block.
        pre_act: bool
          Whether to use pre-activation in the Transpose.
        dropout: float
          Dropout rate.
        padding_mode: str
          Method used for padding.
          Available values are: 'zeros', 'reflect', 'replicate' or 'circular'.
        """
        super(ResBlockTranspose, self).__init__()

        # Using this padding value for kernel with size > 1
        # makes the output channel size irrelevant to the stride.
        p = int((kernel_size - 1) / 2)

        out_channels = int(in_channels / stride)
        self.stride = stride
        self.use_short_cut = use_short_cut

        if not pre_act:
            self.convs = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=p, bias=bias),
                norm(out_channels),
                act(),
                nn.Dropout(dropout),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=p, padding_mode=padding_mode, bias=bias),
                norm(out_channels)
            )
        else:
            self.convs = nn.Sequential(
                norm(in_channels),
                act(),
                nn.Dropout(dropout),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=p, bias=bias),
                norm(out_channels),
                act(),
                nn.Dropout(dropout),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=p, padding_mode=padding_mode, bias=bias),
            )

        if use_short_cut:
            self.shortcut = None
            # shortcut tensor size will be mismatched with the main stream tensor
            if stride != 1:
                if up_sample == "conv":
                    if not pre_act:
                        self.shortcut = nn.Sequential(
                            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias),
                            norm(out_channels)
                        )
                    else:
                        self.shortcut = nn.Sequential(
                            norm(in_channels),
                            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias),
                        )
                else:
                    raise NotImplementedError
            else:
                self.shortcut = Identity()

    def forward(self, x):
        x1 = self.convs(x)
        if self.use_short_cut:
            if self.shortcut is not None:
                x1 += self.shortcut(x)

        return x1


class ResnetDecoder(nn.Module):
    """
    Defines ResNet-like decoder part in a network.
    The input to the decoder is assumed to be a feature map.
    The decoder can be used in typical two ways:
      1) as decoder in GAN with random vector as input. 
         In such case, we need to define several layers to first covert
         the input vector to a feature map.
      2) as decoder in GAN with image as input.
         In such case, we need to define an encoder to covert the input image
         to a feature map.
    """

    def __init__(self, n_blocks_list, stride_list, in_channels, out_channels, kernel_size, norm, act,
                 up_sample, bias, use_short_cut, pre_act, dropout, padding_mode, use_out_act):
        """
        Initialize ResNet based decoder.

        Parameters
        ----------
        n_blocks_list: List(int)
          The list (arbitrary length) specifies number of building blocks in each part.
        stride_list: List(int)
          The list (arbitrary length) specifies stride of building blocks in each part.
          Ensure that len(n_blocks_list) == len(n_blocks_list).
        in_channels: int
          Number of channels of input tensor.
        out_channels: int
          Number of channels of output tensor.
        kernel_size: int
          Kernel size of convolution operator in building block.
        norm: nn.Module
          The normalization method used in the block.
          Set this to "IdentityNorm" to disable normalization.
        act: nn.Module object
          The activation function used in the block.
        up_sample: str
          The up-sampling method used in the block when
          the shortcut tensor size is mismatched with the
          main stream tensor.
          The available values are (only 1x1 conv is supported now):
            1. "conv": use 1x1 conv layer to match the size;
        bias: bool
          Whether to use bias in conv layer.
        use_short_cut: bool
          Whether to use shortcut connection in the block.
          Set this to False make the block as a simply two-layers conv block.
        pre_act: bool
          Whether to use pre-activation in the ResBlock.
        dropout: float
          Dropout rate.
        padding_mode: str
          Method used for padding. 
          Available values are: 'zeros', 'reflect', 'replicate' or 'circular'.
        use_out_act: bool
          Whether to use activation on the output of ResBlock.

        Return
        ------
        convs: nn.Sequential
          A sequence of NN modules.
        """
        super(ResnetDecoder, self).__init__()
        convs = []
        for i, (n_blocks, stride) in enumerate(zip(n_blocks_list, stride_list)):
            convs += ResnetDecoder._make_res_part(in_channels, kernel_size, stride, norm, act, up_sample, bias,
                                                  use_short_cut, n_blocks, pre_act, dropout, padding_mode, use_out_act)
            in_channels = int(in_channels / stride)

        p = int((kernel_size - 1) / 2)
        if not pre_act:
            convs += [
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=p, padding_mode=padding_mode)
            ]
        else:
            convs += [
                norm(in_channels),
                act(),
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=p, padding_mode=padding_mode)
            ]

        convs.append(nn.Tanh())
        convs = nn.Sequential(*convs)
        self.convs = convs

    def forward(self, x):
        return self.convs(x)

    @staticmethod
    def _make_res_part(in_channels, kernel_size, stride, norm, act, up_sample, bias,
                       use_short_cut, n_blocks, pre_act, dropout, padding_mode, use_out_act):
        """
        Utility function for constructing res part in resnet.

        Parameters
        ----------
        in_channels: int
          Number of channels of input tensor.
        kernel_size: int
          Kernel size of convolution operator.
        stride: int
          Stride of the first conv layer in the block.
          The output channel of the first conv layer is: in_channels/stride
        norm: nn.Module
          The normalization method used in the block.
          Set this to "IdentityNorm" to disable normalization.
        act: nn.Module
          The activation function used in the block.
        up_sample: str
          The up-sampling method used in the block when
          the shortcut tensor size is mismatched with the
          main stream tensor.
          The available values are (only 1x1 conv is supported now):
            1. "conv": use 1x1 conv layer to match the size;
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
        padding_mode: str
          Method used for padding.
          Available values are: 'zeros', 'reflect', 'replicate' or 'circular'.
        use_out_act: bool
          Whether to use activation on the output of ResBlockTranspose.

        Return
        ------
        layers: list(ResBlockTranspose)
          A list of ResBlockTranspose.
        """

        # The first res block is with the specified "stride", and
        # all others are with stride=1.
        strides = [stride] + [1] * (n_blocks - 1)

        layers = []
        for i, stride in enumerate(strides):
            layers.append(
                ResBlockTranspose(in_channels, kernel_size, stride, norm, act, up_sample, bias, use_short_cut, pre_act, dropout, padding_mode)
            )
            in_channels = int(in_channels * stride)

            if not pre_act and use_out_act:
                layers.append(act())

        return layers


@register.name_to_model.register("ResNetGenVec")
class ResNetGenVec(nn.Module):
    """
    ResNet-based generator with input of vector.
    This is generally used as generator for GAN.
    """

    def __init__(self, input_dim, hidden_channels, hidden_size, n_blocks_list, stride_list, out_channels,
                 kernel_size, norm, act, up_sample, bias, use_short_cut, pre_act, dropout, padding_mode, use_out_act):
        """
        Initialize ResNet-based generator with input of vector.

        Parameters
        ----------
        input_dim: int
          Dimension of the input vector.
        hidden_channels: int
          Number of channels of the first hidden layer.
        hidden_size: int or tuple
          The size of the first hidden layer.
          This is used to reshape the hidden activation to a feature map with size (hidden_channels, w, h).
          If hidden_size is int, w=h=hidden_size;
          If hidden_size is tuple, (w, h)=hidden_size.
        n_blocks_list: List(int)
          The list (arbitrary length) specifies number of building blocks in each part.
        stride_list: List(int)
          The list (arbitrary length) specifies stride of building blocks in each part.
          Ensure that len(n_blocks_list) == len(n_blocks_list).
        out_channels: int
          Number of channels of output tensor.
        kernel_size: int
          Kernel size of convolution operator in building block.
        norm: nn.Module
          The normalization method used in the block.
          Set this to "IdentityNorm" to disable normalization.
        act: nn.Module object
          The activation function used in the block.
        up_sample: str
          The up-sampling method used in the block when
          the shortcut tensor size is mismatched with the
          main stream tensor.
          The available values are (only 1x1 conv is supported now):
            1. "conv": use 1x1 conv layer to match the size;
        bias: bool
          Whether to use bias in conv layer.
        use_short_cut: bool
          Whether to use shortcut connection in the block.
          Set this to False make the block as a simply two-layers conv block.
        pre_act: bool
          Whether to use pre-activation in the ResBlock.
        dropout: float
          Dropout rate.
        padding_mode: str
          Method used for padding.
          Available values are: 'zeros', 'reflect', 'replicate' or 'circular'.
        use_out_act: bool
          Whether to use activation on the output of ResBlock.
        """
        super(ResNetGenVec, self).__init__()
        if isinstance(hidden_size, int):
            w = h = hidden_size
        elif isinstance(hidden_size, tuple):
            w, h = hidden_size

        self.fc = nn.Linear(input_dim, w * h * hidden_channels)
        self.decoder = ResnetDecoder(n_blocks_list, stride_list, hidden_channels, out_channels,
                                     kernel_size, norm, act, up_sample, bias, use_short_cut,
                                     pre_act, dropout, padding_mode, use_out_act)
        self.w, self.h = w, h
        self.hidden_channels = hidden_channels

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.hidden_channels, self.w, self.h)
        x = self.decoder(x)
        return x

    @staticmethod
    def make_network(configs):
        norm = get_norm(configs["norm"])
        act = get_activation(configs["act"])

        default_params = {
            "input_dim": 128,
            "hidden_channels": 512,
            "hidden_size": 6,
            "out_channels": 3,
            "n_blocks_list": [1, 1, 1],
            "stride_list": [2, 2, 2],
            "kernel_size": 3,
            "norm": norm,
            "act": act,
            "up_sample": "conv",
            "bias": False,
            "use_short_cut": True,
            "pre_act": False,
            "dropout": 0,
            "padding_mode": "reflect",
            "use_out_act": False,
        }

        for key in default_params.keys():
            if key not in ["norm", "act"] and key in configs:
                default_params[key] = configs[key]

        return ResNetGenVec(**default_params)
