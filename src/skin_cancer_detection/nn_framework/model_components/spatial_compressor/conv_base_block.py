import torch
from torch import nn
from collections import OrderedDict
from typing import Literal


class ConvBaseBlock(nn.Module):
    """
    Note: Every instance of this block will compress the spatial dimension, look at _get_reduction_layer.
    """

    def __init__(self, block_num: int, n_input_channels: int,
                 n_layers: int, n_filters_per_layer: list[int],
                 kernel_size: list[int], stride: list[int], padding: list[int] | list[tuple[int, int, int, int]],
                 activation: Literal['hardswish', 'relu'] = 'relu',
                 reduction_strat: Literal['conv', 'max_pool', 'avg_pool'] = 'conv',
                 reduction_kernel: int = 2, reduction_stride: int = 2,
                 reduction_pad: int = None,
                 dtype=None):
        """
        Notes:
            - block_num indicates the sequential position of this block in the model.
        """

        super().__init__()

        self.activations = {
            'relu': nn.ReLU,
            'hardswish': nn.Hardswish
        }

        self.block_num = block_num
        self.n_input_channels = n_input_channels

        self.n_layers = n_layers
        self.n_filters_per_layer = n_filters_per_layer

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self._assure_correct_config(activation, reduction_strat)
        self.activation = self.activations[activation]

        self.reduction_strat = reduction_strat
        self.reduction_kernel = reduction_kernel
        self.reduction_stride = reduction_stride
        self.reduction_pad = reduction_pad

        self.dtype = dtype

        self.block = self.build_conv_block()

    def build_conv_block(self):
        layers = []

        for i in range(self.n_layers):
            in_channels = self.n_filters_per_layer[i - 1] if i != 0 else self.n_input_channels
            layers.append(
                (f'block_{self.block_num}_padding_{i}',
                 nn.ConstantPad2d(self.padding[i], 0))
            )
            layers.append(
                (f'block_{self.block_num}_conv_{i}',
                 nn.Conv2d(
                     in_channels=in_channels,
                     out_channels=self.n_filters_per_layer[i],
                     kernel_size=self.kernel_size[i], stride=self.stride[i],
                     dtype=self.dtype)
                 )
            )
            layers.append((f'block_{self.block_num}_batch_norm_{i}', nn.BatchNorm2d(self.n_filters_per_layer[i])))
            layers.append((f'block_{self.block_num}_activation_{i}', self.activation()))

        # divides len of time dimension by two
        layers.append(self._get_reduction_layer(channels=self.n_filters_per_layer[-1]))
        layers.append((f'block_{self.block_num}_end_block_activation', self.activation()))
        return nn.Sequential(
            OrderedDict(
                layers
            )
        )

    def forward(self, x):
        """Note: PyTorch forward method expects the input to be a batch of samples, even if the batch size is 1."""
        return self.block(x)

    def debug_forward(self, x):
        for name, layer in self.block.named_children():
            print("Name: ", name, " Layer: ", layer)
            x = layer(x)
            print(f'Output shape {x.shape}')
            print()
        return x

    def _get_reduction_layer(self, channels):
        reduction_strats = {
            'conv': (f'block_{self.block_num}_conv_reduce',
                     nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=self.reduction_kernel, stride=self.reduction_stride,
                               padding=self.reduction_pad,
                               dtype=self.dtype)),
            'avg_pool': (f'block_{self.block_num}_avg_pool_reduce',
                         nn.AvgPool2d(kernel_size=self.reduction_kernel, stride=self.reduction_stride)),
            'max_pool': (f'block_{self.block_num}_max_pool_reduce',
                         nn.MaxPool2d(kernel_size=self.reduction_kernel, stride=self.reduction_stride))
        }
        return reduction_strats[self.reduction_strat]

    def _assure_correct_config(self, activation, reduction_strat):
        try:
            assert self.n_layers == len(self.n_filters_per_layer), \
                f"n_layers ({self.n_layers}) != len(n_filters_per_layer) ({len(self.n_filters_per_layer)})"
            assert self.n_layers == len(self.kernel_size), \
                f"n_layers ({self.n_layers}) != len(kernel_size) ({len(self.kernel_size)})"
            assert self.n_layers == len(self.stride), \
                f"n_layers ({self.n_layers}) != len(stride) ({len(self.stride)})"
            assert self.n_layers == len(self.padding), \
                f"n_layers ({self.n_layers}) != len(padding) ({len(self.padding)})"
            assert activation in self.activations, \
                f"activation '{activation}' not in supported activations: {self.activations}"
            assert reduction_strat in ['conv', 'max_pool', 'avg_pool'], \
                f"reduction_strat '{reduction_strat}' must be one of: ['conv', 'max_pool', 'avg_pool']"
        except AssertionError as e:
            message = f"[Config Error in block {self.block_num}] {e}"
            raise AssertionError(message)


if __name__ == "__main__":
    # Usage example
    width = 256
    height = 256
    in_channels = 3

    sample = torch.randn((4, in_channels, width, height))

    model = ConvBaseBlock(block_num=1, n_input_channels=in_channels,
                          n_layers=3, n_filters_per_layer=[16, 64, 128],
                          kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[2, 2, 2],
                          activation='relu',
                          reduction_strat='max_pool',
                          dtype=torch.float32)
    sample = model.debug_forward(sample)
    print("Shape after: ", sample.shape)
