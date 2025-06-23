"""
Comprehensive ML pipeline for skin cancer detection.

This pipeline integrates:
- Data preprocessing
- Model training (image and tabular)
- Hyperparameter optimization
- Model validation and testing
- XAI explanations
- Model deployment preparation
"""

from .pipeline import create_ml_pipeline

__all__ = ["create_ml_pipeline"]