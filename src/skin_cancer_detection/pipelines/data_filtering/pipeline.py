"""
Data filtering pipeline for skin cancer detection project.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import filter_data_node


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the data filtering pipeline.

    Returns:
        A Pipeline object containing the data filtering workflow
    """
    return pipeline(
        [
            node(
                func=filter_data_node,
                inputs=None,
                outputs="filtering_results",
                name="filter_data_node",
            ),
        ]
    )