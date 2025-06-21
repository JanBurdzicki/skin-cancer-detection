"""
CSV joining pipeline for combining CSV files from restructured data.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    join_csv_files_node,
    save_csv_output_node,
    save_parquet_output_node,
    create_visualization_node,
    generate_summary_node,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the CSV joining pipeline.

    This pipeline depends on the csv_label_fixing pipeline to ensure
    CSV labels are properly formatted before joining.

    Returns:
        Kedro Pipeline for CSV joining
    """
    return pipeline(
        [
            node(
                func=join_csv_files_node,
                inputs="parameters",
                outputs="combined_dataframe",
                name="join_csv_files_node",
                tags=["csv_joining", "data_processing"],
            ),
            node(
                func=save_csv_output_node,
                inputs=["combined_dataframe", "parameters"],
                outputs="csv_output_path",
                name="save_csv_output_node",
                tags=["csv_joining", "output"],
            ),
            node(
                func=save_parquet_output_node,
                inputs=["combined_dataframe", "parameters"],
                outputs="parquet_output_path",
                name="save_parquet_output_node",
                tags=["csv_joining", "output"],
            ),
            node(
                func=create_visualization_node,
                inputs=["combined_dataframe", "parameters"],
                outputs="visualization_path",
                name="create_visualization_node",
                tags=["csv_joining", "visualization"],
            ),
            node(
                func=generate_summary_node,
                inputs="combined_dataframe",
                outputs="data_summary",
                name="generate_summary_node",
                tags=["csv_joining", "summary"],
            ),
        ],
        namespace="csv_joining",
        tags=["csv_joining"],
    )