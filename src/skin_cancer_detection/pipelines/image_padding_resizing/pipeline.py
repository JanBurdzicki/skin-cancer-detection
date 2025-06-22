"""
Image padding and resizing pipeline.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import pad_and_resize_images, calculate_optimal_padding_size


def create_pipeline(**kwargs) -> Pipeline:
    """Create the image padding and resizing pipeline.

    Returns:
        A Kedro pipeline for padding and resizing images.
    """
    return pipeline([
        node(
            func=calculate_optimal_padding_size,
            inputs=[
                "params:padding_resizing_input_dir",
                "params:force_square"
            ],
            outputs="padding_dimensions",
            name="calculate_padding_dimensions",
        ),
        node(
            func=pad_and_resize_images,
            inputs=[
                "params:padding_resizing_input_dir",
                "params:padding_resizing_output_dir",
                "params:resize_for_model",
                "params:target_padding_size",
                "params:force_square"
            ],
            outputs="padding_resizing_stats",
            name="pad_and_resize_images",
        ),
    ])