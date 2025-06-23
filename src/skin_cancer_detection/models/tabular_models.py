"""
Tabular models for skin cancer detection.

This module implements various tabular classification models including XGBoost,
LightGBM, Random Forest, and Logistic Regression with comprehensive training
and evaluation capabilities.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
import joblib

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TabularModelConfig:
    """Configuration for tabular models."""

    # Model selection
    model_type: str = "xgboost"  # "xgboost", "lightgbm", "random_forest", "logistic_regression"

    # Data preprocessing
    scale_features: bool = False
    handle_missing: str = "drop"  # "drop", "fill_mean", "fill_median"

    # Training parameters
    test_size: float = 0.2
    random_state: int = 42

    # Model-specific parameters
    xgboost_params: Dict[str, Any] = None
    lightgbm_params: Dict[str, Any] = None
    random_forest_params: Dict[str, Any] = None
    logistic_regression_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.xgboost_params is None:
            self.xgboost_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': 11.27,  # Handle class imbalance
                'random_state': self.random_state
            }

        if self.lightgbm_params is None:
            self.lightgbm_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': 11.27,  # Handle class imbalance
                'random_state': self.random_state,
                'verbose': -1
            }

        if self.random_forest_params is None:
            self.random_forest_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'class_weight': 'balanced',  # Handle class imbalance
                'random_state': self.random_state
            }

        if self.logistic_regression_params is None:
            self.logistic_regression_params = {
                'class_weight': 'balanced',  # Handle class imbalance
                'random_state': self.random_state,
                'max_iter': 1000
            }


class BaseTabularClassifier:
    """Base class for tabular classifiers."""

    def __init__(self, config: TabularModelConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.is_fitted = False

    def _preprocess_data(self, X: pd.DataFrame, y: pd.Series = None, fit_transformers: bool = False):
        """Preprocess the data."""
        X_processed = X.copy()

        # Handle missing values
        if self.config.handle_missing == "drop":
            X_processed = X_processed.dropna()
            if y is not None:
                y = y.loc[X_processed.index]
        elif self.config.handle_missing == "fill_mean":
            X_processed = X_processed.fillna(X_processed.mean())
        elif self.config.handle_missing == "fill_median":
            X_processed = X_processed.fillna(X_processed.median())

        # Store feature names
        if fit_transformers:
            self.feature_names = X_processed.columns.tolist()

        # Scale features if requested
        if self.config.scale_features:
            if fit_transformers:
                self.scaler = StandardScaler()
                X_processed = pd.DataFrame(
                    self.scaler.fit_transform(X_processed),
                    columns=X_processed.columns,
                    index=X_processed.index
                )
            elif self.scaler is not None:
                X_processed = pd.DataFrame(
                    self.scaler.transform(X_processed),
                    columns=X_processed.columns,
                    index=X_processed.index
                )

        # Handle labels
        if y is not None:
            if fit_transformers and y.dtype == 'object':
                self.label_encoder = LabelEncoder()
                y = pd.Series(
                    self.label_encoder.fit_transform(y),
                    index=y.index
                )
            elif self.label_encoder is not None and y.dtype == 'object':
                y = pd.Series(
                    self.label_encoder.transform(y),
                    index=y.index
                )

        return X_processed, y

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the model."""
        X_processed, y_processed = self._preprocess_data(X, y, fit_transformers=True)
        self.model.fit(X_processed, y_processed)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_processed, _ = self._preprocess_data(X, fit_transformers=False)
        predictions = self.model.predict(X_processed)

        # Convert back to original labels if label encoder was used
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)

        return predictions

    def predict_proba(self, X: pd.DataFrame):
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_processed, _ = self._preprocess_data(X, fit_transformers=False)
        return self.model.predict_proba(X_processed)

    def get_feature_importance(self):
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            raise ValueError("Model does not support feature importance")

        return dict(zip(self.feature_names, importance))


class XGBoostClassifier(BaseTabularClassifier):
    """XGBoost classifier for tabular data."""

    def __init__(self, config: TabularModelConfig):
        super().__init__(config)
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
        self.model = xgb.XGBClassifier(**config.xgboost_params)


class LightGBMClassifier(BaseTabularClassifier):
    """LightGBM classifier for tabular data."""

    def __init__(self, config: TabularModelConfig):
        super().__init__(config)
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")
        self.model = lgb.LGBMClassifier(**config.lightgbm_params)


class RandomForestClassifier(BaseTabularClassifier):
    """Random Forest classifier for tabular data."""

    def __init__(self, config: TabularModelConfig):
        super().__init__(config)
        self.model = SKRandomForestClassifier(**config.random_forest_params)


class LogisticRegressionClassifier(BaseTabularClassifier):
    """Logistic Regression classifier for tabular data."""

    def __init__(self, config: TabularModelConfig):
        super().__init__(config)
        self.model = SKLogisticRegression(**config.logistic_regression_params)


class TabularModelTrainer:
    """Trainer for tabular models with comprehensive evaluation."""

    def __init__(self, config: TabularModelConfig):
        self.config = config
        self.model = None
        self.training_history = {}

    def create_model(self):
        """Create model based on configuration."""
        if self.config.model_type == "xgboost":
            return XGBoostClassifier(self.config)
        elif self.config.model_type == "lightgbm":
            return LightGBMClassifier(self.config)
        elif self.config.model_type == "random_forest":
            return RandomForestClassifier(self.config)
        elif self.config.model_type == "logistic_regression":
            return LogisticRegressionClassifier(self.config)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def train(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Train the model."""
        logger.info(f"Training {self.config.model_type} model...")

        # Create model
        self.model = self.create_model()

        # Split data if validation set not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config.test_size,
                random_state=self.config.random_state, stratify=y
            )
        else:
            X_train, y_train = X, y

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate on validation set
        val_predictions = self.model.predict(X_val)
        val_probabilities = self.model.predict_proba(X_val)

        # Calculate metrics
        metrics = self._calculate_metrics(y_val, val_predictions, val_probabilities)

        # Store training history
        self.training_history = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'val_metrics': metrics,
            'feature_importance': self.model.get_feature_importance()
        }

        logger.info(f"Training completed. Validation F1: {metrics['f1']:.4f}")

        return self.model

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Evaluate the model on test data."""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")

        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)

        metrics = self._calculate_metrics(y_test, predictions, probabilities)

        logger.info(f"Test evaluation completed. F1: {metrics['f1']:.4f}")

        return metrics

    def _calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate comprehensive metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

        # Add ROC AUC if probabilities are available
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:  # Multi-class
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except ValueError:
                metrics['roc_auc'] = 0.0

        return metrics

    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")

        model_data = {
            'model': self.model,
            'config': self.config,
            'training_history': self.training_history
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model."""
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.config = model_data['config']
        self.training_history = model_data.get('training_history', {})

        logger.info(f"Model loaded from {filepath}")

        return self.model


def create_tabular_model(model_type: str, **kwargs):
    """Factory function to create tabular models."""
    config = TabularModelConfig(model_type=model_type, **kwargs)
    trainer = TabularModelTrainer(config)
    return trainer.create_model()


# Utility functions for backward compatibility
def get_available_tabular_models():
    """Get list of available tabular model types."""
    available = ['random_forest', 'logistic_regression']

    if XGBOOST_AVAILABLE:
        available.append('xgboost')

    if LIGHTGBM_AVAILABLE:
        available.append('lightgbm')

    return available


def is_model_available(model_type: str):
    """Check if a specific model type is available."""
    return model_type in get_available_tabular_models()