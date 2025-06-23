"""
Unified model training interface for skin cancer detection.

This module provides a unified interface for training both tabular and image models
with comprehensive logging, evaluation, and model management capabilities.
"""

import logging
import os
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import joblib

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .tabular_models import TabularModelTrainer, TabularModelConfig
from .image_models import ImageModelTrainer, ImageModelConfig

logger = logging.getLogger(__name__)


@dataclass
class UnifiedTrainingConfig:
    """Configuration for unified model training."""

    # General settings
    project_name: str = "skin-cancer-detection"
    experiment_name: str = "unified-training"
    output_dir: str = "models"

    # Model selection
    model_type: str = "tabular"  # "tabular", "image", "ensemble"

    # Training settings
    random_state: int = 42
    use_wandb: bool = True
    save_best_model: bool = True

    # Evaluation settings
    eval_metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "precision", "recall"])

    # Tabular model config
    tabular_config: Optional[TabularModelConfig] = None

    # Image model config
    image_config: Optional[ImageModelConfig] = None

    # Ensemble settings
    ensemble_weights: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.tabular_config is None:
            self.tabular_config = TabularModelConfig()

        if self.image_config is None:
            self.image_config = ImageModelConfig()

        if self.ensemble_weights is None:
            self.ensemble_weights = {"tabular": 0.6, "image": 0.4}


class UnifiedModelTrainer:
    """Unified trainer for both tabular and image models."""

    def __init__(self, config: UnifiedTrainingConfig):
        self.config = config
        self.tabular_trainer = None
        self.image_trainer = None
        self.tabular_model = None
        self.image_model = None
        self.ensemble_model = None
        self.training_history = {}

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb if available
        if config.use_wandb and WANDB_AVAILABLE:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                config=self.config.__dict__,
                reinit=True
            )
            logger.info("WandB initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")

    def train_tabular_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Train tabular model."""
        logger.info("Training tabular model...")

        # Create trainer
        self.tabular_trainer = TabularModelTrainer(self.config.tabular_config)

        # Train model
        self.tabular_model = self.tabular_trainer.train(X_train, y_train, X_val, y_val)

        # Log to wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            self._log_tabular_metrics()

        # Save model
        if self.config.save_best_model:
            model_path = self.output_dir / "tabular_model.pkl"
            self.tabular_trainer.save_model(str(model_path))
            logger.info(f"Tabular model saved to {model_path}")

        return self.tabular_model

    def train_image_model(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Train image model."""
        logger.info("Training image model...")

        # Create trainer
        self.image_trainer = ImageModelTrainer(self.config.image_config)

        # Setup callbacks
        callbacks = []

        if self.config.save_best_model:
            checkpoint_callback = ModelCheckpoint(
                dirpath=str(self.output_dir),
                filename="image_model_best",
                monitor="val_f1",
                mode="max",
                save_top_k=1
            )
            callbacks.append(checkpoint_callback)

        early_stopping = EarlyStopping(
            monitor="val_f1",
            patience=10,
            mode="max"
        )
        callbacks.append(early_stopping)

        # Setup logger
        wandb_logger = None
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb_logger = WandbLogger(
                project=self.config.project_name,
                name=f"{self.config.experiment_name}_image"
            )

        # Create PyTorch Lightning trainer
        pl_trainer = pl.Trainer(
            max_epochs=self.config.image_config.max_epochs,
            callbacks=callbacks,
            logger=wandb_logger,
            accelerator="auto",
            devices="auto"
        )

        # Train model
        self.image_model = self.image_trainer.create_model()
        pl_trainer.fit(self.image_model, train_loader, val_loader)

        # Save model
        if self.config.save_best_model:
            model_path = self.output_dir / "image_model.pth"
            torch.save(self.image_model.state_dict(), model_path)
            logger.info(f"Image model saved to {model_path}")

        return self.image_model

    def train_ensemble(self, tabular_data: Tuple[pd.DataFrame, pd.Series],
                      image_data: Tuple[DataLoader, DataLoader]):
        """Train ensemble of tabular and image models."""
        logger.info("Training ensemble model...")

        # Train individual models
        X_train, y_train = tabular_data
        train_loader, val_loader = image_data

        self.train_tabular_model(X_train, y_train)
        self.train_image_model(train_loader, val_loader)

        # Create ensemble
        self.ensemble_model = EnsembleModel(
            tabular_model=self.tabular_model,
            image_model=self.image_model,
            weights=self.config.ensemble_weights
        )

        # Save ensemble
        if self.config.save_best_model:
            ensemble_path = self.output_dir / "ensemble_model.pkl"
            self.save_ensemble(str(ensemble_path))

        return self.ensemble_model

    def evaluate_model(self, model_type: str, test_data: Any):
        """Evaluate trained model."""
        if model_type == "tabular" and self.tabular_trainer is not None:
            X_test, y_test = test_data
            return self.tabular_trainer.evaluate(X_test, y_test)
        elif model_type == "image" and self.image_model is not None:
            test_loader = test_data
            return self._evaluate_image_model(test_loader)
        elif model_type == "ensemble" and self.ensemble_model is not None:
            return self._evaluate_ensemble_model(test_data)
        else:
            raise ValueError(f"Model type {model_type} not trained or not supported")

    def _evaluate_image_model(self, test_loader: DataLoader):
        """Evaluate image model."""
        self.image_model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch
                outputs = self.image_model(images)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='weighted'),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted')
        }

        return metrics

    def _evaluate_ensemble_model(self, test_data: Dict[str, Any]):
        """Evaluate ensemble model."""
        tabular_data = test_data.get('tabular')
        image_data = test_data.get('image')

        if tabular_data is None or image_data is None:
            raise ValueError("Ensemble evaluation requires both tabular and image test data")

        return self.ensemble_model.evaluate(tabular_data, image_data)

    def _log_tabular_metrics(self):
        """Log tabular model metrics to wandb."""
        if not (self.config.use_wandb and WANDB_AVAILABLE):
            return

        metrics = self.tabular_trainer.training_history.get('val_metrics', {})

        # Log metrics
        for metric_name, value in metrics.items():
            wandb.log({f"tabular_val_{metric_name}": value})

        # Log feature importance
        feature_importance = self.tabular_trainer.training_history.get('feature_importance', {})
        if feature_importance:
            # Create feature importance plot
            import matplotlib.pyplot as plt

            features = list(feature_importance.keys())[:10]  # Top 10 features
            importance = [feature_importance[f] for f in features]

            plt.figure(figsize=(10, 6))
            plt.barh(features, importance)
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importance')
            plt.tight_layout()

            wandb.log({"feature_importance": wandb.Image(plt)})
            plt.close()

    def save_ensemble(self, filepath: str):
        """Save ensemble model."""
        if self.ensemble_model is None:
            raise ValueError("No ensemble model to save")

        ensemble_data = {
            'tabular_model': self.tabular_model,
            'image_model_state': self.image_model.state_dict() if self.image_model else None,
            'weights': self.config.ensemble_weights,
            'config': self.config
        }

        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble model saved to {filepath}")

    def load_ensemble(self, filepath: str):
        """Load ensemble model."""
        ensemble_data = joblib.load(filepath)

        self.tabular_model = ensemble_data['tabular_model']

        if ensemble_data['image_model_state'] is not None:
            self.image_model = self.image_trainer.create_model()
            self.image_model.load_state_dict(ensemble_data['image_model_state'])

        self.ensemble_model = EnsembleModel(
            tabular_model=self.tabular_model,
            image_model=self.image_model,
            weights=ensemble_data['weights']
        )

        logger.info(f"Ensemble model loaded from {filepath}")

        return self.ensemble_model

    def get_training_summary(self):
        """Get comprehensive training summary."""
        summary = {
            'config': self.config.__dict__,
            'models_trained': []
        }

        if self.tabular_model is not None:
            summary['models_trained'].append('tabular')
            summary['tabular_history'] = self.tabular_trainer.training_history

        if self.image_model is not None:
            summary['models_trained'].append('image')

        if self.ensemble_model is not None:
            summary['models_trained'].append('ensemble')

        return summary


class EnsembleModel:
    """Ensemble model combining tabular and image predictions."""

    def __init__(self, tabular_model, image_model, weights: Dict[str, float]):
        self.tabular_model = tabular_model
        self.image_model = image_model
        self.weights = weights

    def predict(self, tabular_data: pd.DataFrame, image_data: torch.Tensor):
        """Make ensemble predictions."""
        # Get tabular predictions
        tabular_probs = self.tabular_model.predict_proba(tabular_data)

        # Get image predictions
        self.image_model.eval()
        with torch.no_grad():
            image_outputs = self.image_model(image_data)
            image_probs = torch.softmax(image_outputs, dim=1).cpu().numpy()

        # Combine predictions
        ensemble_probs = (
            self.weights['tabular'] * tabular_probs +
            self.weights['image'] * image_probs
        )

        return np.argmax(ensemble_probs, axis=1)

    def predict_proba(self, tabular_data: pd.DataFrame, image_data: torch.Tensor):
        """Get ensemble prediction probabilities."""
        # Get tabular predictions
        tabular_probs = self.tabular_model.predict_proba(tabular_data)

        # Get image predictions
        self.image_model.eval()
        with torch.no_grad():
            image_outputs = self.image_model(image_data)
            image_probs = torch.softmax(image_outputs, dim=1).cpu().numpy()

        # Combine predictions
        ensemble_probs = (
            self.weights['tabular'] * tabular_probs +
            self.weights['image'] * image_probs
        )

        return ensemble_probs

    def evaluate(self, tabular_data: Tuple[pd.DataFrame, pd.Series],
                image_data: Tuple[torch.Tensor, torch.Tensor]):
        """Evaluate ensemble model."""
        X_tab, y_true = tabular_data
        X_img, _ = image_data

        predictions = self.predict(X_tab, X_img)
        probabilities = self.predict_proba(X_tab, X_img)

        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        metrics = {
            'accuracy': accuracy_score(y_true, predictions),
            'f1': f1_score(y_true, predictions, average='weighted'),
            'precision': precision_score(y_true, predictions, average='weighted'),
            'recall': recall_score(y_true, predictions, average='weighted')
        }

        return metrics


# Utility functions
def create_unified_trainer(model_type: str = "tabular", **kwargs):
    """Factory function to create unified trainer."""
    config = UnifiedTrainingConfig(model_type=model_type, **kwargs)
    return UnifiedModelTrainer(config)


def train_model(model_type: str, train_data: Any, val_data: Any = None, **kwargs):
    """Quick training function."""
    trainer = create_unified_trainer(model_type=model_type, **kwargs)

    if model_type == "tabular":
        X_train, y_train = train_data
        X_val, y_val = val_data if val_data else (None, None)
        return trainer.train_tabular_model(X_train, y_train, X_val, y_val)
    elif model_type == "image":
        train_loader = train_data
        val_loader = val_data
        return trainer.train_image_model(train_loader, val_loader)
    elif model_type == "ensemble":
        return trainer.train_ensemble(train_data, val_data)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_model(model, model_type: str, test_data: Any):
    """Quick evaluation function."""
    if model_type == "tabular":
        X_test, y_test = test_data
        return model.evaluate(X_test, y_test)
    elif model_type == "image":
        # Implement image evaluation
        pass
    elif model_type == "ensemble":
        return model.evaluate(test_data)
    else:
        raise ValueError(f"Unknown model type: {model_type}")