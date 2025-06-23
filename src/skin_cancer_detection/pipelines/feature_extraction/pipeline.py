"""
Feature extraction pipeline.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    feature_extraction_node,
    validate_input_data_node,
    validate_output_data_node
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the feature extraction pipeline.

    Returns:
        A Kedro pipeline for extracting features from cell data.
    """
    return pipeline([
        node(
            func=validate_input_data_node,
            inputs="params:feature_extraction_input_path",
            outputs="input_data_validation_info",
            name="validate_input_data_node",
            tags=["validation", "feature_extraction", "preprocessing"]
        ),
        node(
            func=feature_extraction_node,
            inputs=[
                "params:feature_extraction_input_path",
                "params:feature_extraction_output_path",
                "params:feature_extraction"
            ],
            outputs="feature_extraction_stats",
            name="feature_extraction_node",
            tags=["feature_extraction", "preprocessing", "data_processing"]
        ),
        node(
            func=validate_output_data_node,
            inputs="params:feature_extraction_output_path",
            outputs="output_data_validation_info",
            name="validate_output_data_node",
            tags=["validation", "feature_extraction", "quality_check"]
        )
    ])