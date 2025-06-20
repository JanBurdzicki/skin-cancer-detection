"""
Data restructuring pipeline for skin cancer detection project.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    restructure_raw_data,
    convert_csvs_to_parquet,
    validate_restructured_data
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the data restructuring pipeline.

    Returns:
        A Pipeline object containing the data restructuring workflow
    """
    return pipeline(
        [
            node(
                func=restructure_raw_data,
                inputs=None,
                outputs="restructuring_status",
                name="restructure_raw_data_node",
            ),
            node(
                func=convert_csvs_to_parquet,
                inputs=None,
                outputs="conversion_status",
                name="convert_csvs_to_parquet_node",
            ),
            node(
                func=validate_restructured_data,
                inputs=None,
                outputs="validation_results",
                name="validate_restructured_data_node",
            ),
        ]
    )