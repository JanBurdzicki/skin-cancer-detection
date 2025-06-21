"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from skin_cancer_detection.pipelines import (
    data_restructuring,
    csv_label_fixing,
    csv_joining,
    data_cleanup,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_restructuring_pipeline = data_restructuring.create_pipeline()
    csv_label_fixing_pipeline = csv_label_fixing.create_pipeline()
    csv_joining_pipeline = csv_joining.create_pipeline()
    data_cleanup_pipeline = data_cleanup.create_pipeline()

    return {
        "data_restructuring": data_restructuring_pipeline,
        "csv_label_fixing": csv_label_fixing_pipeline,
        "data_cleanup": data_cleanup_pipeline,
        "csv_joining": csv_joining_pipeline,
        "__default__": data_restructuring_pipeline
        + csv_label_fixing_pipeline
        + data_cleanup_pipeline
        + csv_joining_pipeline
    }
