"""
FastAPI module for skin cancer detection API.

This module provides REST API endpoints for model predictions, health checks,
and model management with comprehensive request/response handling.
"""

from .main import (
    app,
    PredictionResponse,
    ModelInfo,
    load_model,
    load_model_from_local,
    load_model_from_wandb,
    process_image_input,
    process_tabular_input,
    generate_explanations,
    root,
    health_check,
    get_models,
    predict,
    model_paths,
    wandb_models,
    MODELS_DIR
)

__all__ = [
    # FastAPI app
    "app",

    # Models/Schemas
    "PredictionResponse",
    "ModelInfo",

    # Core functions
    "load_model",
    "load_model_from_local",
    "load_model_from_wandb",
    "process_image_input",
    "process_tabular_data",
    "generate_explanations",

    # API endpoints
    "root",
    "health_check",
    "get_models",
    "predict",

    # Configuration
    "model_paths",
    "wandb_models",
    "MODELS_DIR"
]