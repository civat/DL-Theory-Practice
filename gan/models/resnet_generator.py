import torch.nn as nn
import torch.nn.functional as F

import register
from classification import utils


class ResBlockTranspose(nn.Module):
    """Implementation of ResBlock with transposed convolution."""

    def __init__(self, in_channels, out_channels, kernel_size_up, kernel_size_eq, stride, norm, act, up_sample,
                 bias, use_short_cut, pre_act, dropout, padding_mode, output_layer):
        """
        Initialize transposed ResBlock.

        Parameters
        ----------
        in_channels: int
          Number of channels of input tensor.
        out_channels: int
          Number of channels of output tensor.
        kernel_size_up: int or Tuple
          Kernel size of transposed convolution operator used for up-sampling.
        kernel_size_eq: int or Tuple
          Kernel size of convolution operator which does not change the shape of the feature map.
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
          Whether to use pre-activation in the ResBlockTranspose.
        dropout: float
          Dropout rate.
        padding_mode: str
          Method used for padding.
          Available values are: 'zeros', 'reflect', 'replicate' or 'circular'.
        output_layer: bool
          True if the block is used as output layer (so we do not need to add norm layer);
          False otherwise.
        """
        super(ResBlockTranspose, self).__init__()

        # For transposed conv with kernel size being even number, this padding value can
        # make the output's shape 2x larger.
        p = int((kernel_size_up - 1) / 2)

        self.stride = stride
        self.use_short_cut = use_short_cut
        if stride == 1:
            kernel_size_up = kernel_size_eq

        if not pre_act:
            self.convs = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size_up, stride=stride, padding=p, padding_mode=padding_mode, bias=bias),
                norm(out_channels),
                act(),
                nn.Dropout(dropout),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size_eq, stride=1, padding=int(kernel_size_eq // 2), padding_mode=padding_mode, bias=bias),
            ]
            if not output_layer:
                self.convs += [norm(out_channels)]
            self.convs = nn.Sequential(*self.convs)

        else:
            self.convs = nn.Sequential(
                norm(in_channels),
                act(),
                nn.Dropout(dropout),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size_up, stride=stride, padding=p, padding_mode=padding_mode, bias=bias),
                norm(out_channels),
                act(),
                nn.Dropout(dropout),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size_eq, stride=1, padding=int(kernel_size_eq // 2), padding_mode=padding_mode, bias=bias),
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
                self.shortcut = utils.Identity()

    def forward(self, x):
        x1 = self.convs(x)
        if self.use_short_cut:
            if self.shortcut is not None:
                skip = self.shortcut(x)
                if skip.size(-1) != x1.size(-1) or skip.size(-2) != x1.size(-2):
                    skip = F.interpolate(skip, size=(x1.size(-2), x1.size(-1)))
                x1 += skip

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

    def __init__(self, n_blocks_list, stride_list, in_channels, out_channels, kernel_size_up, kernel_size_eq,
                 norm, act, up_sample, bias, use_short_cut, pre_act, dropout, padding_mode, use_out_act, output_conv):
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
        kernel_size_up: int or Tuple
          Kernel size of transposed convolution operator used for up-sampling.
        kernel_size_eq: int or Tuple
          Kernel size of  convolution operator which does not change the shape of the feature map.
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
        output_conv: bool
          Whether to use an additional conv as the output layer.

        Return
        ------
        convs: nn.Sequential
          A sequence of NN modules.
        """
        super(ResnetDecoder, self).__init__()
        assert len(n_blocks_list) == len(stride_list)
        convs = []

        for i in range(len(n_blocks_list) - 1):
            hidden_channels = int(in_channels / stride_list[i])
            convs += ResnetDecoder._make_res_part(in_channels, hidden_channels, kernel_size_up, kernel_size_eq, stride_list[i],
                                                  norm, act, up_sample, bias, use_short_cut, n_blocks_list[i], pre_act, dropout,
                                                  padding_mode, use_out_act, output_layer=False)
            in_channels = hidden_channels

        if output_conv:
            hidden_channels = int(in_channels / stride_list[-1])
            convs += ResnetDecoder._make_res_part(in_channels, hidden_channels, kernel_size_up, kernel_size_eq, stride_list[-1],
                                                  norm, act, up_sample, bias, use_short_cut, n_blocks_list[-1], pre_act, dropout,
                                                  padding_mode, use_out_act, output_layer=False)
            if not pre_act:
                convs += [
                    nn.Conv2d(hidden_channels, out_channels, kernel_size_eq, padding=int(kernel_size_eq // 2), padding_mode=padding_mode, bias=bias)
                ]
            else:
                convs += [
                    norm(in_channels),
                    act(),
                    nn.Conv2d(hidden_channels, out_channels, kernel_size_eq, padding=int(kernel_size_eq // 2), padding_mode=padding_mode, bias=bias)
                ]

        else:
            convs += ResnetDecoder._make_res_part(in_channels, out_channels, kernel_size_up, kernel_size_eq, stride_list[-1],
                                                  norm, act, up_sample, bias, use_short_cut, n_blocks_list[-1], pre_act, dropout,
                                                  padding_mode, use_out_act, output_layer=True)

        convs.append(nn.Tanh())
        convs = nn.Sequential(*convs)
        self.convs = convs

    def forward(self, x):
        return self.convs(x)

    @staticmethod
    def _make_res_part(in_channels, out_channels, kernel_size_up, kernel_size_eq, stride, norm, act, up_sample, bias,
                       use_short_cut, n_blocks, pre_act, dropout, padding_mode, use_out_act, output_layer):
        """
        Utility function for constructing res part in resnet.

        Parameters
        ----------
        in_channels: int
          Number of channels of input tensor.
        out_channels: int
          Number of channels of output tensor.
        kernel_size_up: int or Tuple
          Kernel size of transposed convolution operator used for up-sampling.
        kernel_size_eq: int or Tuple
          Kernel size of convolution operator which does not change the shape of the feature map.
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
        output_layer: bool
          True if the block is used as output layer (so we do not need to add norm layer);
          False otherwise.

        Return
        ------
        layers: list(ResBlockTranspose)
          A list of ResBlockTranspose.
        """

        # The first res block is with the specified "stride", and
        # all others are with stride=1.
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []

        for i in range(len(strides) - 1):
            layers.append(
                ResBlockTranspose(in_channels, out_channels, kernel_size_up, kernel_size_eq, strides[i], norm, act,
                                  up_sample, bias, use_short_cut, pre_act, dropout, padding_mode, output_layer=False))
            in_channels = out_channels
            if not pre_act and use_out_act:
                layers.append(act())

        layers.append(
            ResBlockTranspose(in_channels, out_channels, kernel_size_up, kernel_size_eq, strides[-1], norm, act,
                              up_sample, bias, use_short_cut, pre_act, dropout, padding_mode, output_layer=output_layer))
        if not output_layer and not pre_act and use_out_act:
            layers.append(act())

        return layers


@register.name_to_model.register("ResNetGenVec")
class ResNetGenVec(nn.Module):
    """
    ResNet-based generator with input of vector.
    This is generally used as generator for GAN.
    """

    def __init__(self, input_dim, hidden_channels, hidden_size, n_blocks_list, stride_list, out_channels,
                 kernel_size_up, kernel_size_eq, norm, act, up_sample, bias, use_short_cut, pre_act, dropout, padding_mode, use_out_act, output_conv):
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
        kernel_size_up: int or Tuple
          Kernel size of transposed convolution operator used for up-sampling.
        kernel_size_eq: int or Tuple
          Kernel size of convolution operator which does not change the shape of the feature map.
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
                                     kernel_size_up, kernel_size_eq, norm, act, up_sample, bias, use_short_cut,
                                     pre_act, dropout, padding_mode, use_out_act, output_conv)
        self.w, self.h = w, h
        self.hidden_channels = hidden_channels

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.hidden_channels, self.w, self.h)
        x = self.decoder(x)
        return x

    @staticmethod
    def make_network(configs):
        norm = utils.get_norm(configs["norm"])
        act = utils.get_activation(configs["act"])

        default_params = {
            "input_dim": 128,
            "hidden_channels": 512,
            "hidden_size": 6,
            "out_channels": 3,
            "n_blocks_list": [1, 1, 1],
            "stride_list": [2, 2, 2],
            "kernel_size_up": 4,
            "kernel_size_eq": 3,
            "norm": norm,
            "act": act,
            "up_sample": "conv",
            "bias": False,
            "use_short_cut": True,
            "pre_act": False,
            "dropout": 0,
            "padding_mode": "reflect",
            "use_out_act": False,
            "output_conv": True,
        }

        default_params = utils.set_params(default_params, configs, excluded_keys=["norm", "act"])
        return ResNetGenVec(**default_params)