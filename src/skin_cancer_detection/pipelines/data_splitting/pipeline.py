"""
Data splitting pipeline.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    data_splitting_node,
    validate_input_data_node,
    validate_output_data_node
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data splitting pipeline.

    Returns:
        A Kedro pipeline for splitting data into train/validation/test sets.
    """
    return pipeline([
        node(
            func=validate_input_data_node,
            inputs=[
                "params:data_splitting_feature_path",
                "params:data_splitting_image_dir"
            ],
            outputs="input_splitting_validation_info",
            name="validate_input_splitting_data_node",
            tags=["validation", "data_splitting", "preprocessing"]
        ),
        node(
            func=data_splitting_node,
            inputs=[
                "params:data_splitting_feature_path",
                "params:data_splitting_image_dir",
                "params:data_splitting_output_dir",
                "params:data_splitting"
            ],
            outputs="data_splitting_stats",
            name="data_splitting_node",
            tags=["data_splitting", "preprocessing", "model_preparation"]
        ),
        node(
            func=validate_output_data_node,
            inputs="params:data_splitting_output_dir",
            outputs="output_splitting_validation_info",
            name="validate_output_splitting_data_node",
            tags=["validation", "data_splitting", "quality_check"]
        )
    ])