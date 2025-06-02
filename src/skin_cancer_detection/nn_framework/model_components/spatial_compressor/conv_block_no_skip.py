from skin_cancer_detection.nn_framework.model_components.spatial_compressor.conv_base_block import ConvBaseBlock


class ConvBlockNoSkip(ConvBaseBlock):
    """Class created for structure consistency."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    # Usage example
    import torch

    width = 256
    height = 256
    in_channels = 3

    sample = torch.randn((1, in_channels, width, height))
    model = ConvBlockNoSkip(block_num=1, n_input_channels=in_channels,
                            n_layers=2, n_filters_per_layer=[16, 32],
                            kernel_size=[2, 2], stride=[1, 1], padding=[0, 0],
                            activation='relu',
                            reduction_strat='max_pool',
                            dtype=torch.float32)
    sample = model.debug_forward(sample)
    print("\n STARTING DEBUG 2 \n")
    model2 = ConvBlockNoSkip(block_num=1, n_input_channels=32,
                             n_layers=2, n_filters_per_layer=[48, 64],
                             kernel_size=[2, 2], stride=[1, 1], padding=[0, 0],
                             activation='relu',
                             reduction_strat='max_pool',
                             dtype=torch.float32)
    sample = model2.debug_forward(sample)
    print("Shape after: ", sample.shape)
