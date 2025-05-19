import torch
import torch.nn as nn
from src.skin_cancer_detection.nn_framework.model_components.spatial_compressor.conv_base_block import ConvBaseBlock
from collections import OrderedDict
from typing import Literal


class ConvBlockSkip(ConvBaseBlock):
    def __init__(self, skip_padding: int | tuple[int, int, int, int],
                 skip_strat: Literal['conv', 'max_pool', 'avg_pool'] = 'conv',
                 skip_kernel: int = 2, skip_stride: int = 2,
                 n_skip_channels: int = 64,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.skip_padding = skip_padding
        self.skip_strat = skip_strat
        self.skip_kernel = skip_kernel
        self.skip_stride = skip_stride
        self.n_skip_channels = n_skip_channels

        self.skip_connection = self._get_skip_connection()
        self._assure_correct_skip_shape()

    def forward(self, x):
        out = self.block(x)
        skip_out = self.skip_connection(x)
        return torch.cat((out, skip_out), dim=1)

    def debug_forward(self, x):
        x_cp = x.clone()
        print("SKIP DEBUG:")
        for name, layer in self.skip_connection.named_children():
            print("Name: ", name, " Layer: ", layer)
            x = layer(x)
            print(f'Output shape {x.shape}')
            print()
        print("BLOCK DEBUG:")
        for name, layer in self.block.named_children():
            print("Name: ", name, " Layer: ", layer)
            x_cp = layer(x_cp)
            print(f'Output shape {x.shape}')
            print()

    def _get_skip_connection(self):
        skip_strats = {
            'conv': (f'block_{self.block_num}_conv_skip',
                     nn.Conv2d(in_channels=self.n_input_channels,
                               out_channels=self.n_skip_channels,
                               kernel_size=self.skip_kernel, stride=self.skip_stride,
                               dtype=self.dtype)),
            'avg_pool': (f'block_{self.block_num}_avg_pool_skip',
                         nn.AvgPool2d(kernel_size=self.skip_kernel, stride=self.skip_stride)),
            'max_pool': (f'block_{self.block_num}_max_pool_skip',
                         nn.MaxPool2d(kernel_size=self.skip_kernel, stride=self.skip_stride))
        }
        return nn.Sequential(
            OrderedDict(
                [
                    (f'block_{self.block_num}_skip_padding',
                     nn.ConstantPad2d(self.skip_padding, 0)),
                    skip_strats[self.skip_strat]
                ]
            )
        )

    def _assure_correct_skip_shape(self):
        x = torch.randn((2, self.n_input_channels, 1000, 1000), dtype=self.dtype)
        x_block = self.block(x)
        x_skip = self.skip_connection(x)
        try:
            assert x_block.shape[-2:] == x_skip.shape[-2:]
        except AssertionError as e:
            message = (f'{e} block number: {self.block_num}. \n Spatial skip connection out shape {x_skip.shape} and'
                       f' block out shape {x_block.shape} are expected to be the same for '
                       f'mocked input data of shape {x.shape}')
            raise AssertionError(message)


if __name__ == "__main__":
    # Usage example
    import torch

    width = 256
    height = 256
    in_channels = 3

    sample = torch.randn((1, in_channels, width, height))

    model = ConvBlockSkip(block_num=1, n_input_channels=in_channels,
                          n_layers=3, n_filters_per_layer=[16, 64, 128],
                          kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=[1, 0, 0],
                          activation='relu',
                          reduction_strat='max_pool', reduction_kernel=3, reduction_stride=2, reduction_pad=0,
                          skip_strat='conv', skip_kernel=3, skip_stride=2, skip_padding=0,
                          dtype=torch.float32)

    sample = model.debug_forward(sample)
