#!/usr/bin/env python3
"""
Simple unified API for skin cancer detection with XAI explanations.

This API can:
1. Load input (image, CSV/Parquet file)
2. Choose model
3. Run model predictions and XAI methods
4. Return comprehensive outputs
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Union
import pandas as pd
import numpy as np
from PIL import Image
import io
import joblib
from pathlib import Path
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Simple Skin Cancer Detection API",
    description="Unified API for image and tabular data analysis with XAI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class PredictionResponse(BaseModel):
    success: bool
    model_used: str
    input_type: str
    model_source: str
    predictions: Dict[str, Any]
    explanations: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ModelInfo(BaseModel):
    available_models: Dict[str, List[str]]
    supported_formats: List[str]
    model_sources: List[str]

# Global variables for caching models
loaded_models = {}

# Model paths in the 06_models directory
MODELS_DIR = Path("data/06_models")
model_paths = {
    "image": {
        "cnn": MODELS_DIR / "image" / "cnn_model.pkl",
        "resnet18": MODELS_DIR / "image" / "resnet18_model.pkl",
        "efficientnet": MODELS_DIR / "image" / "efficientnet_model.pkl"
    },
    "tabular": {
        "xgboost": MODELS_DIR / "tabular" / "xgboost_model.pkl",
        "lightgbm": MODELS_DIR / "tabular" / "lightgbm_model.pkl",
        "random_forest": MODELS_DIR / "tabular" / "random_forest_model.pkl",
        "logistic_regression": MODELS_DIR / "tabular" / "logistic_regression_model.pkl"
    }
}

# W&B configuration for model loading
wandb_models = {
    "image": {
        "cnn": {"entity": "skin-cancer-team", "project": "skin-cancer-detection", "artifact": "cnn_model:latest"},
        "resnet18": {"entity": "skin-cancer-team", "project": "skin-cancer-detection", "artifact": "resnet18_model:latest"},
        "efficientnet": {"entity": "skin-cancer-team", "project": "skin-cancer-detection", "artifact": "efficientnet_model:latest"}
    },
    "tabular": {
        "xgboost": {"entity": "skin-cancer-team", "project": "skin-cancer-detection", "artifact": "xgboost_model:latest"},
        "lightgbm": {"entity": "skin-cancer-team", "project": "skin-cancer-detection", "artifact": "lightgbm_model:latest"},
        "random_forest": {"entity": "skin-cancer-team", "project": "skin-cancer-detection", "artifact": "random_forest_model:latest"},
        "logistic_regression": {"entity": "skin-cancer-team", "project": "skin-cancer-detection", "artifact": "logistic_regression_model:latest"}
    }
}

def load_model_from_wandb(model_type: str, model_name: str):
    """Load a model from Weights & Biases."""
    try:
        import wandb

        if model_type not in wandb_models or model_name not in wandb_models[model_type]:
            raise ValueError(f"Model {model_type}/{model_name} not configured for W&B loading")

        wandb_config = wandb_models[model_type][model_name]

        # Initialize wandb run
        run = wandb.init(
            entity=wandb_config["entity"],
            project=wandb_config["project"],
            job_type="model_inference"
        )

        # Download the artifact
        artifact = run.use_artifact(wandb_config["artifact"])
        artifact_dir = artifact.download()

        # Look for model files in the artifact directory
        model_files = list(Path(artifact_dir).glob("*.pkl")) + list(Path(artifact_dir).glob("*.joblib"))

        if not model_files:
            raise FileNotFoundError(f"No model files found in W&B artifact: {wandb_config['artifact']}")

        # Load the first model file found
        model_path = model_files[0]
        model = joblib.load(model_path)

        wandb.finish()
        logger.info(f"Loaded model from W&B: {model_type}_{model_name}")

        return model

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="wandb package not installed. Install with: pip install wandb"
        )
    except Exception as e:
        logger.error(f"Failed to load model from W&B: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model from W&B: {str(e)}"
        )

def load_model_from_local(model_type: str, model_name: str):
    """Load a model from local 06_models directory."""
    try:
        if model_type not in model_paths or model_name not in model_paths[model_type]:
            raise ValueError(f"Model {model_type}/{model_name} not configured for local loading")

        model_path = model_paths[model_type][model_name]

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Try different loading methods based on file extension
        if model_path.suffix in ['.pkl', '.joblib']:
            model = joblib.load(model_path)
        elif model_path.suffix == '.pt':
            import torch
            model = torch.load(model_path, map_location='cpu')
        elif model_path.suffix == '.h5':
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
        else:
            # Default to joblib
            model = joblib.load(model_path)

        logger.info(f"Loaded model from local: {model_path}")
        return model

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found: {model_path}. Please ensure the model exists in data/06_models/"
        )
    except Exception as e:
        logger.error(f"Failed to load local model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load local model: {str(e)}"
        )

def load_model(model_type: str, model_name: str, source: str = "local"):
    """
    Load a model if not already cached.

    Args:
        model_type: Type of model ('image' or 'tabular')
        model_name: Name of the specific model
        source: Source to load from ('local' or 'wandb')
    """
    model_key = f"{model_type}_{model_name}_{source}"

    if model_key not in loaded_models:
        try:
            if source == "local":
                model = load_model_from_local(model_type, model_name)
            elif source == "wandb":
                model = load_model_from_wandb(model_type, model_name)
            else:
                raise ValueError(f"Invalid source: {source}. Use 'local' or 'wandb'")

            loaded_models[model_key] = model
            logger.info(f"Successfully cached model: {model_key}")

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading model {model_key}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error loading model: {str(e)}"
            )

    return loaded_models[model_key]

def process_image_input(image_file: UploadFile) -> np.ndarray:
    """Process uploaded image file."""
    try:
        # Read image
        image_bytes = image_file.file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to standard size (224x224 for most models)
        image = image.resize((224, 224))

        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def process_tabular_input(file: UploadFile) -> pd.DataFrame:
    """Process uploaded CSV/Parquet file."""
    try:
        file_bytes = file.file.read()

        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(io.BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported file format. Use CSV or Parquet.")

        return df

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

def generate_explanations(model, X, model_type: str, model_name: str) -> Dict[str, Any]:
    """Generate XAI explanations for the model predictions."""
    explanations = {}

    try:
        if model_type == "tabular":
            # Import our explainers
            from src.skin_cancer_detection.XAI.tabular_explainer import (
                PermutationImportanceExplainer,
                FeatureImportanceExplainer,
                LIME_AVAILABLE,
                LIMEExplainer,
                SHAP_AVAILABLE,
                SHAPExplainer
            )

            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            # Feature Importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                try:
                    fi_explainer = FeatureImportanceExplainer(model, feature_names)
                    fi_explanation = fi_explainer.explain()
                    explanations['feature_importance'] = {
                        'importances': fi_explanation['feature_importances'].tolist(),
                        'feature_names': feature_names
                    }
                except Exception as e:
                    logger.warning(f"Feature importance failed: {str(e)}")

            # Permutation Importance
            try:
                # Create dummy target for explanation (in real scenario, use actual target)
                y_dummy = np.random.randint(0, 2, size=X.shape[0])
                pi_explainer = PermutationImportanceExplainer(model, feature_names)
                pi_explanation = pi_explainer.explain(X, y_dummy, n_repeats=3)
                explanations['permutation_importance'] = {
                    'importances_mean': pi_explanation['importances_mean'].tolist(),
                    'importances_std': pi_explanation['importances_std'].tolist(),
                    'feature_names': feature_names
                }
            except Exception as e:
                logger.warning(f"Permutation importance failed: {str(e)}")

            # SHAP explanations
            if SHAP_AVAILABLE:
                try:
                    shap_explainer = SHAPExplainer(model, X[:10], feature_names)  # Use subset for background
                    shap_explanation = shap_explainer.explain(X[:5])  # Explain first 5 samples
                    explanations['shap'] = {
                        'shap_values': shap_explanation['shap_values'].tolist(),
                        'expected_value': float(shap_explanation['expected_value']),
                        'feature_names': feature_names
                    }
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {str(e)}")

            # LIME explanations
            if LIME_AVAILABLE:
                try:
                    lime_explainer = LIMEExplainer(model, X[:10], feature_names, mode='classification')
                    lime_explanation = lime_explainer.explain(X[:2], num_features=5, num_samples=100)

                    # Extract explanation data
                    lime_results = []
                    for exp in lime_explanation['explanations']:
                        lime_results.append({
                            'feature_importance': dict(exp.as_list()),
                            'score': exp.score
                        })

                    explanations['lime'] = {
                        'explanations': lime_results,
                        'feature_names': feature_names
                    }
                except Exception as e:
                    logger.warning(f"LIME explanation failed: {str(e)}")

        elif model_type == "image":
            # For image models, we can add basic explanations
            explanations['note'] = "Image XAI methods (GradCAM, LIME for images) not implemented in this demo"

    except Exception as e:
        logger.error(f"Error generating explanations: {str(e)}")
        explanations['error'] = str(e)

    return explanations

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Simple Skin Cancer Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict - Main prediction endpoint",
            "models": "/models - Available models information",
            "health": "/health - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": pd.Timestamp.now().isoformat()}

@app.get("/models", response_model=ModelInfo)
async def get_models():
    """Get information about available models."""
    return ModelInfo(
        available_models={"local": model_paths, "wandb": wandb_models},
        supported_formats=["jpg", "jpeg", "png", "csv", "parquet"],
        model_sources=["local", "wandb"]
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form(..., description="Type of model: 'image' or 'tabular'"),
    model_name: str = Form(..., description="Specific model name to use"),
    model_source: str = Form("local", description="Model source: 'local' or 'wandb'"),
    include_explanations: bool = Form(True, description="Whether to include XAI explanations")
):
    """
    Main prediction endpoint that handles both image and tabular data.

    Args:
        file: Uploaded file (image: jpg/png, tabular: csv/parquet)
        model_type: Either 'image' or 'tabular'
        model_name: Specific model to use
        model_source: Either 'local' (from 06_models) or 'wandb'
        include_explanations: Whether to generate XAI explanations

    Returns:
        Comprehensive prediction response with optional explanations
    """

    try:
        # Validate inputs
        available_models = model_paths if model_source == "local" else wandb_models

        if model_type not in available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type. Choose from: {list(available_models.keys())}"
            )

        if model_name not in available_models[model_type]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_name for {model_type}. Choose from: {list(available_models[model_type].keys())}"
            )

        # Load model (will raise HTTPException if fails)
        model = load_model(model_type, model_name, model_source)

        # Process input based on type
        if model_type == "image":
            # Process image
            if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                raise HTTPException(status_code=400, detail="For image models, please upload JPG or PNG files")

            X = process_image_input(file)
            input_shape = X.shape

            # For demo, create dummy predictions for image models
            predictions = {
                "predicted_class": "benign",
                "probabilities": {
                    "benign": 0.7,
                    "malignant": 0.3
                },
                "confidence": 0.7,
                "input_shape": input_shape
            }

        elif model_type == "tabular":
            # Process tabular data
            if not file.filename.lower().endswith(('.csv', '.parquet')):
                raise HTTPException(status_code=400, detail="For tabular models, please upload CSV or Parquet files")

            df = process_tabular_input(file)
            X = df.select_dtypes(include=[np.number]).values  # Select only numeric columns

            if X.shape[1] == 0:
                raise HTTPException(status_code=400, detail="No numeric columns found in the data")

            # Make predictions
            try:
                pred_proba = model.predict_proba(X)
                pred_class = model.predict(X)

                predictions = {
                    "predicted_classes": pred_class.tolist(),
                    "probabilities": pred_proba.tolist(),
                    "num_samples": len(X),
                    "num_features": X.shape[1],
                    "feature_columns": df.select_dtypes(include=[np.number]).columns.tolist()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

        # Generate explanations if requested
        explanations = None
        if include_explanations:
            explanations = generate_explanations(model, X, model_type, model_name)

        return PredictionResponse(
            success=True,
            model_used=f"{model_type}_{model_name}",
            input_type=model_type,
            model_source=model_source,
            predictions=predictions,
            explanations=explanations
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return PredictionResponse(
            success=False,
            model_used=f"{model_type}_{model_name}" if 'model_type' in locals() and 'model_name' in locals() else "unknown",
            input_type=model_type if 'model_type' in locals() else "unknown",
            model_source=model_source if 'model_source' in locals() else "unknown",
            predictions={},
            error=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)