"""
Model deployment utilities for skin cancer detection.

This module provides functionality for model deployment, inference, and serving
including model loading, preprocessing, prediction, and API integration.
"""

import logging
import os
import pickle
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import joblib

try:
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""

    # Model paths
    tabular_model_path: str = "models/tabular_model.pkl"
    image_model_path: str = "models/image_model.pth"
    ensemble_model_path: str = "models/ensemble_model.pkl"

    # Deployment settings
    model_type: str = "tabular"  # "tabular", "image", "ensemble"
    batch_size: int = 32
    max_batch_size: int = 100

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "Skin Cancer Detection API"
    api_version: str = "1.0.0"

    # Image preprocessing
    image_size: Tuple[int, int] = (224, 224)
    normalize_mean: List[float] = None
    normalize_std: List[float] = None

    # Feature settings
    expected_features: List[str] = None
    feature_scaling: bool = False

    # Monitoring
    log_predictions: bool = True
    use_wandb_monitoring: bool = False

    def __post_init__(self):
        if self.normalize_mean is None:
            self.normalize_mean = [0.485, 0.456, 0.406]
        if self.normalize_std is None:
            self.normalize_std = [0.229, 0.224, 0.225]


class ModelDeployer:
    """Base class for model deployment."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        self.prediction_count = 0

        # Setup logging
        self._setup_logging()

        # Initialize monitoring
        if config.use_wandb_monitoring and WANDB_AVAILABLE:
            self._init_wandb_monitoring()

    def _setup_logging(self):
        """Setup prediction logging."""
        if self.config.log_predictions:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            # Create prediction logger
            pred_logger = logging.getLogger("predictions")
            handler = logging.FileHandler(log_dir / "predictions.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            pred_logger.addHandler(handler)
            pred_logger.setLevel(logging.INFO)

            self.prediction_logger = pred_logger

    def _init_wandb_monitoring(self):
        """Initialize WandB monitoring."""
        try:
            wandb.init(
                project="skin-cancer-deployment",
                name=f"deployment-{self.config.model_type}",
                config=self.config.__dict__
            )
            logger.info("WandB monitoring initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB monitoring: {e}")

    def load_model(self):
        """Load the model for deployment."""
        raise NotImplementedError("Subclasses must implement load_model")

    def preprocess(self, data: Any):
        """Preprocess input data."""
        raise NotImplementedError("Subclasses must implement preprocess")

    def predict(self, data: Any):
        """Make predictions."""
        raise NotImplementedError("Subclasses must implement predict")

    def postprocess(self, predictions: Any):
        """Postprocess predictions."""
        return predictions

    def predict_batch(self, batch_data: List[Any]):
        """Make batch predictions."""
        if len(batch_data) > self.config.max_batch_size:
            raise ValueError(f"Batch size {len(batch_data)} exceeds maximum {self.config.max_batch_size}")

        results = []
        for data in batch_data:
            try:
                result = self.predict(data)
                results.append(result)
            except Exception as e:
                logger.error(f"Prediction failed for batch item: {e}")
                results.append({"error": str(e)})

        return results

    def _log_prediction(self, input_data: Any, prediction: Any, confidence: float = None):
        """Log prediction for monitoring."""
        if not self.config.log_predictions:
            return

        self.prediction_count += 1

        log_data = {
            "prediction_id": self.prediction_count,
            "model_type": self.config.model_type,
            "prediction": prediction,
            "confidence": confidence
        }

        if hasattr(self, 'prediction_logger'):
            self.prediction_logger.info(f"Prediction: {log_data}")

        # Log to WandB if available
        if self.config.use_wandb_monitoring and WANDB_AVAILABLE:
            wandb.log(log_data)


class TabularModelDeployer(ModelDeployer):
    """Deployer for tabular models."""

    def load_model(self):
        """Load tabular model."""
        model_path = Path(self.config.tabular_model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            # Load model data
            model_data = joblib.load(model_path)

            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.config.expected_features = getattr(
                    self.model, 'feature_names', None
                )
            else:
                self.model = model_data

            self.is_loaded = True
            logger.info(f"Tabular model loaded from {model_path}")

        except Exception as e:
            logger.error(f"Failed to load tabular model: {e}")
            raise

    def preprocess(self, data: Union[Dict, pd.DataFrame]):
        """Preprocess tabular data."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Data must be dict or DataFrame")

        # Check for expected features
        if self.config.expected_features:
            missing_features = set(self.config.expected_features) - set(df.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Fill missing features with default values
                for feature in missing_features:
                    df[feature] = 0

            # Reorder columns to match training
            df = df[self.config.expected_features]

        # Handle missing values
        df = df.fillna(0)

        return df

    def predict(self, data: Union[Dict, pd.DataFrame]):
        """Make tabular prediction."""
        if not self.is_loaded:
            self.load_model()

        # Preprocess
        processed_data = self.preprocess(data)

        try:
            # Make prediction
            prediction = self.model.predict(processed_data)[0]

            # Get prediction probabilities if available
            confidence = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_data)[0]
                confidence = float(np.max(probabilities))

            # Convert prediction to standard format
            if isinstance(prediction, (np.integer, np.floating)):
                prediction = int(prediction)
            elif isinstance(prediction, str):
                prediction = prediction

            result = {
                "prediction": prediction,
                "confidence": confidence,
                "model_type": "tabular"
            }

            # Log prediction
            self._log_prediction(data, prediction, confidence)

            return result

        except Exception as e:
            logger.error(f"Tabular prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


class ImageModelDeployer(ModelDeployer):
    """Deployer for image models."""

    def load_model(self):
        """Load image model."""
        model_path = Path(self.config.image_model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            # Load model architecture (assuming it's defined elsewhere)
            from .image_models import SimpleImageModel

            self.model = SimpleImageModel(num_classes=2)
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()

            self.is_loaded = True
            logger.info(f"Image model loaded from {model_path}")

        except Exception as e:
            logger.error(f"Failed to load image model: {e}")
            raise

    def preprocess(self, image: Union[Image.Image, np.ndarray, torch.Tensor]):
        """Preprocess image data."""
        try:
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            elif isinstance(image, torch.Tensor):
                image = Image.fromarray(image.numpy().astype(np.uint8))

            # Resize image
            image = image.resize(self.config.image_size)

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert to tensor and normalize
            image_array = np.array(image) / 255.0
            image_tensor = torch.from_numpy(image_array).float()
            image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW

            # Normalize
            mean = torch.tensor(self.config.normalize_mean).view(3, 1, 1)
            std = torch.tensor(self.config.normalize_std).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std

            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)

            return image_tensor

        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise

    def predict(self, image: Union[Image.Image, np.ndarray]):
        """Make image prediction."""
        if not self.is_loaded:
            self.load_model()

        # Preprocess
        processed_image = self.preprocess(image)

        try:
            with torch.no_grad():
                # Make prediction
                outputs = self.model(processed_image)
                probabilities = F.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1).item()
                confidence = float(torch.max(probabilities))

            result = {
                "prediction": prediction,
                "confidence": confidence,
                "model_type": "image",
                "probabilities": probabilities.squeeze().tolist()
            }

            # Log prediction
            self._log_prediction("image_data", prediction, confidence)

            return result

        except Exception as e:
            logger.error(f"Image prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


class EnsembleModelDeployer(ModelDeployer):
    """Deployer for ensemble models."""

    def __init__(self, config: DeploymentConfig):
        super().__init__(config)
        self.tabular_deployer = TabularModelDeployer(config)
        self.image_deployer = ImageModelDeployer(config)
        self.ensemble_weights = {"tabular": 0.6, "image": 0.4}

    def load_model(self):
        """Load ensemble model."""
        # Load individual models
        self.tabular_deployer.load_model()
        self.image_deployer.load_model()

        # Load ensemble weights if available
        ensemble_path = Path(self.config.ensemble_model_path)
        if ensemble_path.exists():
            try:
                ensemble_data = joblib.load(ensemble_path)
                self.ensemble_weights = ensemble_data.get('weights', self.ensemble_weights)
                logger.info(f"Ensemble weights loaded: {self.ensemble_weights}")
            except Exception as e:
                logger.warning(f"Failed to load ensemble weights: {e}")

        self.is_loaded = True
        logger.info("Ensemble model loaded successfully")

    def predict(self, tabular_data: Dict, image_data: Image.Image):
        """Make ensemble prediction."""
        if not self.is_loaded:
            self.load_model()

        try:
            # Get individual predictions
            tabular_result = self.tabular_deployer.predict(tabular_data)
            image_result = self.image_deployer.predict(image_data)

            # Combine predictions
            tabular_conf = tabular_result.get('confidence', 0.5)
            image_conf = image_result.get('confidence', 0.5)

            # Weighted confidence
            ensemble_confidence = (
                self.ensemble_weights['tabular'] * tabular_conf +
                self.ensemble_weights['image'] * image_conf
            )

            # Simple majority vote for now
            if tabular_result['prediction'] == image_result['prediction']:
                ensemble_prediction = tabular_result['prediction']
            else:
                # Use the model with higher confidence
                if tabular_conf > image_conf:
                    ensemble_prediction = tabular_result['prediction']
                else:
                    ensemble_prediction = image_result['prediction']

            result = {
                "prediction": ensemble_prediction,
                "confidence": ensemble_confidence,
                "model_type": "ensemble",
                "individual_predictions": {
                    "tabular": tabular_result,
                    "image": image_result
                }
            }

            # Log prediction
            self._log_prediction(
                {"tabular": tabular_data, "image": "image_data"},
                ensemble_prediction,
                ensemble_confidence
            )

            return result

        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# API Creation Functions
def create_tabular_api(config: DeploymentConfig):
    """Create FastAPI app for tabular model."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")

    app = FastAPI(
        title=config.api_title,
        version=config.api_version,
        description="Tabular model API for skin cancer detection"
    )

    deployer = TabularModelDeployer(config)

    @app.on_event("startup")
    async def startup_event():
        deployer.load_model()

    @app.post("/predict")
    async def predict(data: Dict[str, Any]):
        try:
            result = deployer.predict(data)
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict_batch")
    async def predict_batch(data: List[Dict[str, Any]]):
        try:
            results = deployer.predict_batch(data)
            return JSONResponse(content={"predictions": results})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "model_loaded": deployer.is_loaded}

    return app


def create_image_api(config: DeploymentConfig):
    """Create FastAPI app for image model."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")

    app = FastAPI(
        title=config.api_title,
        version=config.api_version,
        description="Image model API for skin cancer detection"
    )

    deployer = ImageModelDeployer(config)

    @app.on_event("startup")
    async def startup_event():
        deployer.load_model()

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        try:
            # Read image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))

            result = deployer.predict(image)
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "model_loaded": deployer.is_loaded}

    return app


# Utility functions
def create_deployer(model_type: str, config: DeploymentConfig = None):
    """Factory function to create model deployer."""
    if config is None:
        config = DeploymentConfig(model_type=model_type)

    if model_type == "tabular":
        return TabularModelDeployer(config)
    elif model_type == "image":
        return ImageModelDeployer(config)
    elif model_type == "ensemble":
        return EnsembleModelDeployer(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def deploy_model(model_type: str, config: DeploymentConfig = None):
    """Quick deployment function."""
    if config is None:
        config = DeploymentConfig(model_type=model_type)

    if model_type == "tabular":
        app = create_tabular_api(config)
    elif model_type == "image":
        app = create_image_api(config)
    else:
        raise ValueError(f"API deployment not supported for model type: {model_type}")

    # Run the API
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level="info"
    )


def load_and_predict(model_type: str, model_path: str, data: Any):
    """Quick prediction function."""
    config = DeploymentConfig(model_type=model_type)

    if model_type == "tabular":
        config.tabular_model_path = model_path
        deployer = TabularModelDeployer(config)
    elif model_type == "image":
        config.image_model_path = model_path
        deployer = ImageModelDeployer(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    deployer.load_model()
    return deployer.predict(data)