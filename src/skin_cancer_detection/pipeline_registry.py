"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from skin_cancer_detection.pipelines import (
    data_processing,
    data_restructuring,
    model_training,
    model_evaluation,
    csv_label_fixing,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = data_processing.create_pipeline()
    data_restructuring_pipeline = data_restructuring.create_pipeline()
    model_training_pipeline = model_training.create_pipeline()
    model_evaluation_pipeline = model_evaluation.create_pipeline()
    csv_label_fixing_pipeline = csv_label_fixing.create_pipeline()

    return {
        "data_processing": data_processing_pipeline,
        "data_restructuring": data_restructuring_pipeline,
        "model_training": model_training_pipeline,
        "model_evaluation": model_evaluation_pipeline,
        "csv_label_fixing": csv_label_fixing_pipeline,
        "__default__": data_processing_pipeline
        + data_restructuring_pipeline
        + model_training_pipeline
        + model_evaluation_pipeline
        + csv_label_fixing_pipeline,
    }
