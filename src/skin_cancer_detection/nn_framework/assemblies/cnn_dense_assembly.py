from skin_cancer_detection.nn_framework.model_components.spatial_compressor.conv_block_skip import ConvBlockSkip
from skin_cancer_detection.nn_framework.model_components.spatial_compressor.conv_block_no_skip import \
    ConvBlockNoSkip
from skin_cancer_detection.nn_framework.model_components.classifier.base_classifier import BaseClassifier
import torch
import torch.nn as nn
from typing import Literal
from collections import OrderedDict


class CnnDenseAssembly(nn.Module):
    def __init__(self, block_configs: list[dict],
                 block_types: list[Literal['skip', 'no_skip']],
                 classifier_config: dict,
                 classifier_type: Literal['base'],
                 input_width: int = 256,
                 input_height: int = 256
                 ):
        """
        - block_configs (list[dict]): List of dictionaries containing parameters for each conv block.
        - block_types (list[Literal['skip', 'no_skip']]): Selects class that corresponding config will be passed to.
        - classifier_config (dict): Dictionary containing parameters for classifier.
        - input_width (int): Used for assuring that the assembly's config is correct.
        - input_height (int): Used for assuring that the assembly's config is correct.

        Notes:
            - block_configs and classifier_config will be passed as kwargs e.g: BaseClassifier(**classifier_config).
        """
        super().__init__()

        self.create_block = {
            'no_skip': ConvBlockNoSkip,
            'skip': ConvBlockSkip
        }

        self.create_classifier = {
            'base': BaseClassifier
        }

        self.block_configs = block_configs
        self.block_types = block_types
        self.classifier_config = classifier_config
        self.classifier_type = classifier_type

        self._assure_correct_convolution_config()
        self.conv = self.build_conv()

        self._in_w = input_width
        self._in_h = input_height

        self._assure_correct_classifier_config()
        self.classifier = self.create_classifier['base'](**classifier_config)

    def forward(self, x):
        x_conv = self.conv(x)
        x_flattened = torch.flatten(x_conv, start_dim=1)
        out = self.classifier(x_flattened)
        return out

    def build_conv(self):
        blocks = []
        for i in range(len(self.block_configs)):
            block_cfg = self.block_configs[i]
            block_type = self.block_types[i]
            blocks.append(
                (f'conv_block_{i}', self.create_block[block_type](**block_cfg))
            )
        return nn.Sequential(
            OrderedDict(
                blocks
            )
        )

    def _assure_correct_classifier_config(self):
        x = torch.randn((2, self.block_configs[0]['n_input_channels'], self._in_h, self._in_w),
                        dtype=self.classifier_config.get('dtype', None))

        x_conv = self.conv(x)
        x_sample_after_flatten_shape = torch.flatten(x_conv, start_dim=1).shape[-1]
        n_input_features_classifier = self.classifier_config['n_input_features']
        assert x_sample_after_flatten_shape == n_input_features_classifier, \
            "Number of classifier input features doesn't match sample after flattening"

    def _assure_correct_convolution_config(self):
        """Put assertion rules for module here."""
        assert len(self.block_configs) == len(self.block_types)

        for i in range(1, len(self.block_configs)):
            previous_block_n_out_channels = self.block_configs[i - 1]['n_filters_per_layer'][-1]
            if self.block_types[i-1] == 'skip':
                # 64 is default skip channels number
                previous_block_n_out_channels += self.block_configs[i-1].get('n_skip_channels', 64)

            current_block_n_input_channels = self.block_configs[i]["n_input_channels"]

            previous_block_dtype = self.block_configs[i - 1].get('dtype', None)
            current_block_dtype = self.block_configs[i].get('dtype', None)

            assert previous_block_n_out_channels == current_block_n_input_channels, \
                f'Number of channels outputted by block {i - 1} is expected to be the same as the' \
                f'number of channels outputted by block {i}; \n' \
                f'Number of channels outputted by block {i - 1} is:' \
                f' {self.block_configs[i - 1]["n_filters_per_layer"][-1]}; \n' \
                f'Number of channels taken by block {i} is:' \
                f' {self.block_configs[i]["n_filters_per_layer"][0]}' \

            assert previous_block_dtype == current_block_dtype, \
                f"Dtype of block {i-1} doesn't match dtype of block {i}"

        classifier_dtype = self.classifier_config.get('dtype', None)
        assert current_block_dtype == classifier_dtype, "Classifier dtype doesn't match convolution dtype"


if __name__ == "__main__":
    # Usage example

    width = 256
    height = 256
    in_channels = 3

    block_1_cfg = {
        "block_num": 1,
        "n_input_channels": in_channels,
        "n_layers": 2,
        "n_filters_per_layer": [16, 32],
        "kernel_size": [2, 2],
        "stride": [1, 1],
        "padding": [0, 0],
        "activation": "relu",
        "reduction_strat": "max_pool",
        "skip_padding": 0,
        "n_skip_channels": 64,
        "skip_kernel": 3,
        "skip_stride": 2,
        "dtype": torch.float32,
    }

    block_2_cfg = {
        "block_num": 2,
        "n_input_channels": 96,
        "n_layers": 2,
        "n_filters_per_layer": [48, 64],
        "kernel_size": [2, 2],
        "stride": [1, 1],
        "padding": [0, 0],
        "activation": "relu",
        "reduction_strat": "max_pool",
        "dtype": torch.float32,
    }

    classifier_cfg = {
        "n_layers": 2,
        "n_input_features": 64*62*62,
        "units_per_layer": [512, 256],
        "n_output_values": 42,
        "activation": "hardswish",
        "dtype": torch.float32,
    }

    assembly = CnnDenseAssembly(block_configs=[block_1_cfg, block_2_cfg],
                                block_types=['skip', 'no_skip'],
                                classifier_config=classifier_cfg,
                                classifier_type='base',
                                input_width=width, input_height=height)

    sample = torch.randn((2, in_channels, width, height), dtype=torch.float32)

    output = assembly(sample)
    print(output.shape)
