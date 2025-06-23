"""
Comprehensive ML pipeline for skin cancer detection.

This pipeline integrates:
- Data preprocessing
- Model training (ALL tabular models: RF, XGBoost, LGBM and image models: CNN, ResNet)
- Hyperparameter optimization
- Model validation and testing
- ALL XAI explanations for images
- Model deployment preparation
"""

from .pipeline import (
    create_ml_pipeline,
    create_comprehensive_ml_pipeline,
    create_comprehensive_tabular_training_pipeline,
    create_comprehensive_image_training_pipeline,
    create_comprehensive_evaluation_pipeline,
    create_comprehensive_xai_pipeline,
    create_model_comparison_pipeline,
    create_deployment_pipeline,
    create_tabular_training_pipeline,
    create_image_training_pipeline
)

__all__ = [
    "create_ml_pipeline",
    "create_comprehensive_ml_pipeline",
    "create_comprehensive_tabular_training_pipeline",
    "create_comprehensive_image_training_pipeline",
    "create_comprehensive_evaluation_pipeline",
    "create_comprehensive_xai_pipeline",
    "create_model_comparison_pipeline",
    "create_deployment_pipeline",
    "create_tabular_training_pipeline",
    "create_image_training_pipeline"
]