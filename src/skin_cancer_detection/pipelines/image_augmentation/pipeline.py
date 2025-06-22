"""
Image augmentation pipeline.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    augment_images_for_training,
    create_fluorescent_microscopy_augmentation_config,
    validate_augmentation_results,
    analyze_augmentation_parameters
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the image augmentation pipeline.

    Returns:
        A Kedro pipeline for augmenting images for training.
    """
    return pipeline([
        node(
            func=create_fluorescent_microscopy_augmentation_config,
            inputs=None,
            outputs="augmentation_config",
            name="create_augmentation_config",
        ),
        node(
            func=augment_images_for_training,
            inputs=[
                "params:augmentation_input_dir",
                "params:augmentation_output_dir",
                "params:num_augmentations_per_image",
                "augmentation_config",
                "params:simple_naming",
                "params:save_original",
                "params:augmentation_seed"
            ],
            outputs="augmentation_stats",
            name="augment_images_node",
        ),
        node(
            func=validate_augmentation_results,
            inputs="augmentation_stats",
            outputs="augmentation_validation",
            name="validate_augmentation",
        ),
        node(
            func=analyze_augmentation_parameters,
            inputs="augmentation_stats",
            outputs="augmentation_analysis",
            name="analyze_augmentation_params",
        ),
    ])