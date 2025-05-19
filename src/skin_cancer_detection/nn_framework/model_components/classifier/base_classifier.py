import torch
from torch import nn
from collections import OrderedDict
from typing import Literal


class BaseClassifier(nn.Module):
    def __init__(self, n_layers: int, n_input_features: int,
                 units_per_layer: list[int], n_output_values: int,
                 activation: Literal['relu', 'hardswish'],
                 dtype: torch.dtype = None,
                 sigmoid_output: bool = True):
        """
        - n_layers (int): The number of layers in the model.
          This does not include the final layer, which outputs the probabilities for the classes.
        - n_input_features (int): The number of input features for the first layer.
        - units_per_layer (list[int]): A list specifying the number of units(out_features) in each layer.
        - n_classes (int): The number of output classes for the classifier.
        - activation (str): Which activation function use in between layers.
        - dtype (torch.dtype): Data type of layers.
        - sigmoid_output (bool): Whether to pass final layer output through sigmoid.
        """

        super().__init__()

        self.n_layers = n_layers
        self.n_input_features = n_input_features
        self.units_per_layer = units_per_layer
        self.n_output_values = n_output_values

        self.dtype = dtype
        self.sigmoid_output = sigmoid_output

        self.activations = {
            'relu': nn.ReLU,
            'hardswish': nn.Hardswish
        }

        self._assure_correct_config(activation)
        self.activation = self.activations[activation]

        self.block = self.build()

    def build(self):
        layers = []

        for i in range(self.n_layers):
            in_features = self.n_input_features if i == 0 else self.units_per_layer[i-1]
            layers.append(
                (f"dense_layer_{i}",
                 nn.Linear(
                     in_features=in_features,
                     out_features=self.units_per_layer[i],
                     dtype=self.dtype
                 )
                 )
            )
            layers.append((f"batch_norm_classifier_{i}", nn.BatchNorm1d(self.units_per_layer[i], dtype=self.dtype)))
            layers.append(
                (f"classifier_activation_{i}", self.activation())
            )

        in_features = self.n_input_features if self.n_layers == 0 else self.units_per_layer[self.n_layers - 1]
        layers.append(
            ("dense_classifier",
             nn.Linear(in_features=in_features,
                       out_features=self.n_output_values,
                       dtype=self.dtype)
             )
        )
        layers.append(("batch_norm_classifier_end", nn.BatchNorm1d(self.n_output_values, dtype=self.dtype)))
        if self.sigmoid_output:
            layers.append(
                ("classifier_end_activation", nn.Sigmoid())
            )

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
            # TODO: Logger zamiast printów
            print("Name: ", name, " Layer: ", layer)
            print(f'Contains NaNs before layer: {torch.isnan(x).any()}')
            x = layer(x)
            print(f'Output shape {x.shape}')
            print(f'Contains NaNs after layer: {torch.isnan(x).any()}')
            print()
        return x

    def _assure_correct_config(self, activation):
        """Put assertion rules for module here."""
        assert len(self.units_per_layer) == self.n_layers, \
                f"Length of units_per_layer doesn't match number of layers."
        assert activation in self.activations, \
            f"activation '{activation}' not in supported activations: {self.activations}."


if __name__ == "__main__":
    # Usage example
    n_layers = 3
    n_input_features = 10
    units_per_layer = [32, 64, 128]
    n_output_values = 5

    model = BaseClassifier(
        n_layers=n_layers,
        n_input_features=n_input_features,
        units_per_layer=units_per_layer,
        n_output_values=n_output_values,
        activation='hardswish',
        dtype=torch.float32
    )

    # batch of 4 samples with `n_input_features` per sample
    dummy_input = torch.randn((4, n_input_features), dtype=torch.float32)

    output = model.debug_forward(dummy_input)