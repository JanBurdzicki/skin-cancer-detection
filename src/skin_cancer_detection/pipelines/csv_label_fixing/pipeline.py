"""
CSV label fixing pipeline.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import fix_csv_labels_node


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the CSV label fixing pipeline.

    Returns:
        A Pipeline object containing the CSV label fixing nodes
    """
    return pipeline(
        [
            node(
                func=fix_csv_labels_node,
                inputs=None,
                outputs="csv_label_fixing_result",
                name="fix_csv_labels_node",
            ),
        ],
        namespace="csv_label_fixing",
        inputs={},
        outputs={"csv_label_fixing_result"},
    )