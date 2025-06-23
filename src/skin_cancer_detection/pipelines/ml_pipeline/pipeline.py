"""
Comprehensive ML pipeline for skin cancer detection.

This pipeline provides a complete machine learning workflow using pre-split data:
- Training ALL tabular models (RF, XGBoost, LGBM) and image models (CNN, ResNet)
- Hyperparameter optimization using Optuna
- Model validation and testing
- ALL XAI explanations generation for image models
- Model performance analysis and comparison
- Deployment preparation
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    train_all_tabular_models,
    train_all_image_models,
    train_tabular_model,  # Legacy support
    train_image_model,    # Legacy support
    optimize_tabular_hyperparameters,
    optimize_image_hyperparameters,
    evaluate_tabular_model,
    evaluate_image_model,
    compare_models,
    generate_tabular_explanations,
    generate_comprehensive_image_explanations,
    generate_image_explanations,  # Legacy support
    prepare_model_for_deployment,
    create_model_artifacts
)


def create_comprehensive_tabular_training_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for training ALL tabular models (RF, XGBoost, LGBM)."""
    return pipeline([
        node(
            func=train_all_tabular_models,
            inputs=["tabular_train_data", "tabular_val_data", "params:tabular_models"],
            outputs="all_tabular_trained_models",
            name="train_all_tabular_models_node",
        ),
        node(
            func=optimize_tabular_hyperparameters,
            inputs=["tabular_train_data", "tabular_val_data", "params:optimization"],
            outputs="tabular_optimized_params",
            name="optimize_tabular_hyperparameters_node",
        ),
        node(
            func=train_all_tabular_models,
            inputs=["tabular_train_data", "tabular_val_data", "tabular_optimized_params"],
            outputs="all_tabular_optimized_models",
            name="train_all_optimized_tabular_models_node",
        ),
    ])


def create_comprehensive_image_training_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for training ALL image models (CNN, ResNet)."""
    return pipeline([
        node(
            func=train_all_image_models,
            inputs=["image_train_data", "image_val_data", "params:image_models"],
            outputs="all_image_trained_models",
            name="train_all_image_models_node",
        ),
        node(
            func=optimize_image_hyperparameters,
            inputs=["image_train_data", "image_val_data", "params:optimization"],
            outputs="image_optimized_params",
            name="optimize_image_hyperparameters_node",
        ),
        node(
            func=train_all_image_models,
            inputs=["image_train_data", "image_val_data", "image_optimized_params"],
            outputs="all_image_optimized_models",
            name="train_all_optimized_image_models_node",
        ),
    ])


def create_comprehensive_evaluation_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for evaluating all trained models."""
    return pipeline([
        # Evaluate tabular models (using the best model from the collection)
        node(
            func=lambda models: models['models'][models['best_model']],
            inputs="all_tabular_optimized_models",
            outputs="best_tabular_model",
            name="extract_best_tabular_model_node",
        ),
        node(
            func=evaluate_tabular_model,
            inputs=["best_tabular_model", "tabular_test_data"],
            outputs="tabular_evaluation_results",
            name="evaluate_tabular_model_node",
        ),

        # Evaluate image models (using the best model from the collection)
        node(
            func=lambda models: models['models'][models['best_model']],
            inputs="all_image_optimized_models",
            outputs="best_image_model",
            name="extract_best_image_model_node",
        ),
        node(
            func=evaluate_image_model,
            inputs=["best_image_model", "image_test_data"],
            outputs="image_evaluation_results",
            name="evaluate_image_model_node",
        ),
    ])


def create_comprehensive_xai_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for generating ALL XAI explanations."""
    return pipeline([
        # Generate tabular explanations
        node(
            func=generate_tabular_explanations,
            inputs=["best_tabular_model", "tabular_test_data", "params:xai"],
            outputs="tabular_explanations",
            name="generate_tabular_explanations_node",
        ),

        # Generate comprehensive image explanations with ALL XAI methods
        node(
            func=generate_comprehensive_image_explanations,
            inputs=["all_image_optimized_models", "image_test_data", "params:xai"],
            outputs="comprehensive_image_explanations",
            name="generate_comprehensive_image_explanations_node",
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

        # Compare all tabular models internally
        node(
            func=lambda models: models['comparison'],
            inputs="all_tabular_optimized_models",
            outputs="tabular_models_comparison",
            name="tabular_models_comparison_node",
        ),

        # Compare all image models internally
        node(
            func=lambda models: models['comparison'],
            inputs="all_image_optimized_models",
            outputs="image_models_comparison",
            name="image_models_comparison_node",
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
                "comprehensive_image_explanations",
                "params:artifacts"
            ],
            outputs="model_artifacts",
            name="create_model_artifacts_node",
        ),
    ])


def create_comprehensive_ml_pipeline(**kwargs) -> Pipeline:
    """Create the comprehensive ML pipeline with ALL models and XAI methods."""

    # All sub-pipelines
    tabular_pipeline = create_comprehensive_tabular_training_pipeline()
    image_pipeline = create_comprehensive_image_training_pipeline()
    evaluation_pipeline = create_comprehensive_evaluation_pipeline()
    xai_pipeline = create_comprehensive_xai_pipeline()
    comparison_pipeline = create_model_comparison_pipeline()
    deployment_pipeline = create_deployment_pipeline()

    # Combine all pipelines
    return (
        tabular_pipeline +
        image_pipeline +
        evaluation_pipeline +
        xai_pipeline +
        comparison_pipeline +
        deployment_pipeline
    )


# Legacy pipeline for backward compatibility
def create_tabular_training_pipeline(**kwargs) -> Pipeline:
    """Legacy: Create pipeline for single tabular model training."""
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
    """Legacy: Create pipeline for single image model training."""
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


def create_ml_pipeline(**kwargs) -> Pipeline:
    """Legacy: Create the original ML pipeline."""
    tabular_pipeline = create_tabular_training_pipeline()
    image_pipeline = create_image_training_pipeline()
    comparison_pipeline = create_model_comparison_pipeline()
    deployment_pipeline = create_deployment_pipeline()

    return (
        tabular_pipeline +
        image_pipeline +
        comparison_pipeline +
        deployment_pipeline
    )