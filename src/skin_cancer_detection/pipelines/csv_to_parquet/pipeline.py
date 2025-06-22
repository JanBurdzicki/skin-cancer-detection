"""
CSV to Parquet conversion pipeline for skin cancer detection project.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import convert_csvs_to_parquet


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the CSV to Parquet conversion pipeline.

    Returns:
        A Pipeline object containing the CSV to Parquet conversion workflow
    """
    return pipeline(
        [
            node(
                func=convert_csvs_to_parquet,
                inputs=None,
                outputs="conversion_status",
                name="convert_csvs_to_parquet_node",
            ),
        ]
    )