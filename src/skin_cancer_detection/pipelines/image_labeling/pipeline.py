"""
Image labeling pipeline.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    image_labeling_node,
    validate_label_data_node
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the image labeling pipeline.

    Returns:
        A Kedro pipeline for labeling images with gfp_status.
    """
    return pipeline([
        node(
            func=validate_label_data_node,
            inputs="params:label_csv_path",
            outputs="label_validation_info",
            name="validate_label_data_node",
            tags=["validation", "image_labeling"]
        ),
        node(
            func=image_labeling_node,
            inputs=[
                "params:labeling_input_dir",
                "params:labeling_output_dir",
                "params:label_csv_path"
            ],
            outputs="image_labeling_stats",
            name="image_labeling_node",
            tags=["image_labeling", "preprocessing"]
        )
    ])