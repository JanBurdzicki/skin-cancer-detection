"""
Data cleanup pipeline definition.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import cleanup_invalid_data_dry_run_node, cleanup_invalid_data_execute_node


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the data cleanup pipeline.

    Returns:
        A Pipeline instance for data cleanup
    """
    return pipeline(
        [
            node(
                func=cleanup_invalid_data_dry_run_node,
                inputs="params:invalid_patterns_file",
                outputs=None,
                name="cleanup_invalid_data_dry_run_node",
                tags=["cleanup", "dry_run"],
            ),
            node(
                func=cleanup_invalid_data_execute_node,
                inputs="params:invalid_patterns_file",
                outputs=None,
                name="cleanup_invalid_data_execute_node",
                tags=["cleanup", "execute"],
            ),
        ]
    )