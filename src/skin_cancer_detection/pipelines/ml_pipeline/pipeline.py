"""
Simplified ML pipeline for skin cancer detection.

This pipeline provides a streamlined machine learning workflow using pre-split data:
- Model training for both image and tabular data
- Hyperparameter optimization using Optuna
- Model validation and testing
- XAI explanations generation
- Model performance analysis
- Deployment preparation
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    train_tabular_model,
    train_image_model,
    optimize_tabular_hyperparameters,
    optimize_image_hyperparameters,
    evaluate_tabular_model,
    evaluate_image_model,
    compare_models,
    generate_tabular_explanations,
    generate_image_explanations,
    prepare_model_for_deployment,
    create_model_artifacts
)


def create_tabular_training_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for tabular model training and optimization."""
    return pipeline([
        node(
            func=train_tabular_model,
            inputs=["tabular_train_data", "tabular_val_data", "params:tabular_model"],
            outputs="tabular_trained_model",
            name="train_tabular_model_node",
        ),
        node(
            func=optimize_tabular_hyperparameters,
            inputs=["tabular_train_data", "tabular_val_data", "params:optimization"],
            outputs="tabular_optimized_params",
            name="optimize_tabular_hyperparameters_node",
        ),
        node(
            func=train_tabular_model,
            inputs=["tabular_train_data", "tabular_val_data", "tabular_optimized_params"],
            outputs="tabular_optimized_model",
            name="train_optimized_tabular_model_node",
        ),
        node(
            func=evaluate_tabular_model,
            inputs=["tabular_optimized_model", "tabular_test_data"],
            outputs="tabular_evaluation_results",
            name="evaluate_tabular_model_node",
        ),
        node(
            func=generate_tabular_explanations,
            inputs=["tabular_optimized_model", "tabular_test_data", "params:xai"],
            outputs="tabular_explanations",
            name="generate_tabular_explanations_node",
        ),
    ])


def create_image_training_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for image model training and optimization."""
    return pipeline([
        node(
            func=train_image_model,
            inputs=["image_train_data", "image_val_data", "params:image_model"],
            outputs="image_trained_model",
            name="train_image_model_node",
        ),
        node(
            func=optimize_image_hyperparameters,
            inputs=["image_train_data", "image_val_data", "params:optimization"],
            outputs="image_optimized_params",
            name="optimize_image_hyperparameters_node",
        ),
        node(
            func=train_image_model,
            inputs=["image_train_data", "image_val_data", "image_optimized_params"],
            outputs="image_optimized_model",
            name="train_optimized_image_model_node",
        ),
        node(
            func=evaluate_image_model,
            inputs=["image_optimized_model", "image_test_data"],
            outputs="image_evaluation_results",
            name="evaluate_image_model_node",
        ),
        node(
            func=generate_image_explanations,
            inputs=["image_optimized_model", "image_test_data", "params:xai"],
            outputs="image_explanations",
            name="generate_image_explanations_node",
        ),
    ])


def create_model_comparison_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for comparing different models."""
    return pipeline([
        node(
            func=compare_models,
            inputs=[
                "tabular_evaluation_results",
                "image_evaluation_results",
                "params:model_comparison"
            ],
            outputs="model_comparison_results",
            name="compare_models_node",
        ),
    ])


def create_deployment_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for preparing models for deployment."""
    return pipeline([
        node(
            func=prepare_model_for_deployment,
            inputs=[
                "tabular_evaluation_results",
                "image_evaluation_results",
                "model_comparison_results",
                "params:deployment"
            ],
            outputs="deployment_ready_model",
            name="prepare_model_for_deployment_node",
        ),
        node(
            func=create_model_artifacts,
            inputs=[
                "deployment_ready_model",
                "tabular_explanations",
                "image_explanations",
                "params:artifacts"
            ],
            outputs="model_artifacts",
            name="create_model_artifacts_node",
        ),
    ])


def create_ml_pipeline(**kwargs) -> Pipeline:
    """Create the simplified ML pipeline."""
    tabular_pipeline = create_tabular_training_pipeline()
    image_pipeline = create_image_training_pipeline()
    comparison_pipeline = create_model_comparison_pipeline()
    deployment_pipeline = create_deployment_pipeline()

    # Combine all pipelines
    return (
        tabular_pipeline +
        image_pipeline +
        comparison_pipeline +
        deployment_pipeline
    )