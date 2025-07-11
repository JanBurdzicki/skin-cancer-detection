"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from skin_cancer_detection.pipelines import (
    data_restructuring,
    csv_label_fixing,
    csv_to_parquet,
    data_cleanup,
    csv_joining,
    data_filtering,
    image_padding_resizing,
    image_labeling,
    image_augmentation,
    feature_extraction,
    data_splitting,
    ml_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Individual component pipelines
    data_restructuring_pipeline = data_restructuring.create_pipeline()
    csv_label_fixing_pipeline = csv_label_fixing.create_pipeline()
    csv_to_parquet_pipeline = csv_to_parquet.create_pipeline()
    data_cleanup_pipeline = data_cleanup.create_pipeline()
    csv_joining_pipeline = csv_joining.create_pipeline()
    data_filtering_pipeline = data_filtering.create_pipeline()
    image_padding_resizing_pipeline = image_padding_resizing.create_pipeline()
    image_labeling_pipeline = image_labeling.create_pipeline()
    image_augmentation_pipeline = image_augmentation.create_pipeline()
    feature_extraction_pipeline = feature_extraction.create_pipeline()
    data_splitting_pipeline = data_splitting.create_pipeline()

    # ML Pipelines
    # Comprehensive ML pipeline with ALL models (RF, XGBoost, LGBM, CNN, ResNet) and ALL XAI methods
    comprehensive_ml_pipeline = ml_pipeline.create_comprehensive_ml_pipeline()

    # Legacy ML pipeline (single model training)
    legacy_ml_pipeline = ml_pipeline.create_ml_pipeline()

    # Individual ML sub-pipelines
    tabular_training_pipeline = ml_pipeline.create_comprehensive_tabular_training_pipeline()
    image_training_pipeline = ml_pipeline.create_comprehensive_image_training_pipeline()
    evaluation_pipeline = ml_pipeline.create_comprehensive_evaluation_pipeline()
    xai_pipeline = ml_pipeline.create_comprehensive_xai_pipeline()
    comparison_pipeline = ml_pipeline.create_model_comparison_pipeline()
    deployment_pipeline = ml_pipeline.create_deployment_pipeline()

    return {
        # Individual component pipelines
        "data_restructuring": data_restructuring_pipeline,
        "csv_label_fixing": csv_label_fixing_pipeline,
        "csv_to_parquet": csv_to_parquet_pipeline,
        "data_cleanup": data_cleanup_pipeline,
        "csv_joining": csv_joining_pipeline,
        "data_filtering": data_filtering_pipeline,
        "image_padding_resizing": image_padding_resizing_pipeline,
        "image_labeling": image_labeling_pipeline,
        "image_augmentation": image_augmentation_pipeline,
        "feature_extraction": feature_extraction_pipeline,
        "data_splitting": data_splitting_pipeline,

        # Comprehensive ML pipeline (default) - Trains ALL models with ALL XAI methods
        "__default__": comprehensive_ml_pipeline,
        "comprehensive_ml": comprehensive_ml_pipeline,
        "ml_pipeline": comprehensive_ml_pipeline,  # Main pipeline alias

        # Legacy ML pipeline (single model training)
        "legacy_ml": legacy_ml_pipeline,

        # Individual ML sub-pipelines for granular execution
        "tabular_training": tabular_training_pipeline,    # Trains RF, XGBoost, LGBM
        "image_training": image_training_pipeline,        # Trains CNN, ResNet
        "model_evaluation": evaluation_pipeline,          # Evaluates all models
        "xai_explanations": xai_pipeline,                # Generates ALL XAI methods
        "model_comparison": comparison_pipeline,          # Compares model performance
        "model_deployment": deployment_pipeline,          # Prepares for deployment

        # Complete data processing pipeline
        "data_processing": data_restructuring_pipeline
        + csv_label_fixing_pipeline
        + csv_to_parquet_pipeline
        + data_cleanup_pipeline
        + csv_joining_pipeline
        + data_filtering_pipeline
        + image_padding_resizing_pipeline
        + image_labeling_pipeline
        + image_augmentation_pipeline
        + feature_extraction_pipeline
        + data_splitting_pipeline,

        # Full end-to-end pipeline (data processing + comprehensive ML)
        "full_pipeline": data_restructuring_pipeline
        + csv_label_fixing_pipeline
        + csv_to_parquet_pipeline
        + data_cleanup_pipeline
        + csv_joining_pipeline
        + data_filtering_pipeline
        + image_padding_resizing_pipeline
        + image_labeling_pipeline
        + image_augmentation_pipeline
        + feature_extraction_pipeline
        + data_splitting_pipeline
        + comprehensive_ml_pipeline
    }
