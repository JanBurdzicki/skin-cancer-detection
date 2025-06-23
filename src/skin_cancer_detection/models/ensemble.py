"""
Ensemble methods for skin cancer detection.

This module implements various ensemble techniques for combining predictions
from multiple models including voting, stacking, and weighted averaging.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""

    # Ensemble type
    ensemble_type: str = "weighted_average"  # "voting", "stacking", "weighted_average", "adaptive"

    # Voting settings
    voting_method: str = "soft"  # "hard", "soft"

    # Weighting settings
    weights: Dict[str, float] = None
    adaptive_weights: bool = False
    weight_update_frequency: int = 100

    # Stacking settings
    meta_learner: str = "logistic_regression"
    use_cross_validation: bool = True
    cv_folds: int = 5

    # Performance tracking
    track_individual_performance: bool = True
    performance_window: int = 1000

    # Model selection
    min_models_required: int = 2
    max_models_allowed: int = 10

    def __post_init__(self):
        if self.weights is None:
            self.weights = {}


class BaseEnsemble:
    """Base class for ensemble methods."""

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.models = {}
        self.weights = config.weights.copy()
        self.performance_history = {}
        self.prediction_count = 0
        self.is_fitted = False

    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """Add a model to the ensemble."""
        if len(self.models) >= self.config.max_models_allowed:
            raise ValueError(f"Maximum number of models ({self.config.max_models_allowed}) reached")

        self.models[name] = model
        if name not in self.weights:
            self.weights[name] = weight

        # Initialize performance tracking
        if self.config.track_individual_performance:
            self.performance_history[name] = []

        logger.info(f"Added model '{name}' to ensemble with weight {weight}")

    def remove_model(self, name: str):
        """Remove a model from the ensemble."""
        if name in self.models:
            del self.models[name]
            if name in self.weights:
                del self.weights[name]
            if name in self.performance_history:
                del self.performance_history[name]
            logger.info(f"Removed model '{name}' from ensemble")

    def get_model_names(self):
        """Get list of model names in ensemble."""
        return list(self.models.keys())

    def fit(self, X: Any, y: Any):
        """Fit the ensemble."""
        if len(self.models) < self.config.min_models_required:
            raise ValueError(f"At least {self.config.min_models_required} models required")

        # Fit individual models if they have a fit method
        for name, model in self.models.items():
            if hasattr(model, 'fit'):
                try:
                    model.fit(X, y)
                    logger.info(f"Fitted model '{name}'")
                except Exception as e:
                    logger.warning(f"Failed to fit model '{name}': {e}")

        self.is_fitted = True
        return self

    def predict(self, X: Any):
        """Make ensemble predictions."""
        raise NotImplementedError("Subclasses must implement predict method")

    def predict_proba(self, X: Any):
        """Get ensemble prediction probabilities."""
        raise NotImplementedError("Subclasses must implement predict_proba method")

    def _normalize_weights(self):
        """Normalize weights to sum to 1."""
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for name in self.weights:
                self.weights[name] /= total_weight

    def _update_adaptive_weights(self, predictions: Dict[str, Any], true_labels: Any):
        """Update weights based on recent performance."""
        if not self.config.adaptive_weights:
            return

        # Calculate individual model accuracies
        for name, pred in predictions.items():
            if name in self.models:
                accuracy = accuracy_score(true_labels, pred)
                self.performance_history[name].append(accuracy)

                # Keep only recent performance
                if len(self.performance_history[name]) > self.config.performance_window:
                    self.performance_history[name] = self.performance_history[name][-self.config.performance_window:]

                # Update weight based on recent performance
                recent_performance = np.mean(self.performance_history[name][-100:])  # Last 100 predictions
                self.weights[name] = max(0.1, recent_performance)  # Minimum weight of 0.1

        # Normalize weights
        self._normalize_weights()

        logger.debug(f"Updated adaptive weights: {self.weights}")


class VotingEnsemble(BaseEnsemble):
    """Voting ensemble implementation."""

    def predict(self, X: Any):
        """Make predictions using voting."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")

        predictions = {}

        # Get predictions from all models
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                    predictions[name] = pred
                else:
                    logger.warning(f"Model '{name}' does not have predict method")
            except Exception as e:
                logger.warning(f"Prediction failed for model '{name}': {e}")

        if not predictions:
            raise ValueError("No valid predictions obtained from models")

        # Combine predictions
        if self.config.voting_method == "hard":
            return self._hard_voting(predictions)
        else:
            return self._soft_voting(predictions, X)

    def _hard_voting(self, predictions: Dict[str, np.ndarray]):
        """Hard voting - majority vote."""
        # Stack predictions
        pred_array = np.array(list(predictions.values()))

        # Apply weights if using weighted voting
        if any(w != 1.0 for w in self.weights.values()):
            weighted_votes = []
            for i, (name, pred) in enumerate(predictions.items()):
                weight = self.weights.get(name, 1.0)
                # Repeat predictions based on weight (rounded)
                repeat_count = max(1, int(round(weight * 10)))
                weighted_votes.extend([pred] * repeat_count)
            pred_array = np.array(weighted_votes)

        # Majority vote
        from scipy import stats
        ensemble_pred, _ = stats.mode(pred_array, axis=0, keepdims=False)
        return ensemble_pred

    def _soft_voting(self, predictions: Dict[str, np.ndarray], X: Any):
        """Soft voting - average probabilities."""
        probabilities = {}

        # Get probabilities from all models
        for name, model in self.models.items():
            if name in predictions:
                try:
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X)
                        probabilities[name] = prob
                    else:
                        # Convert hard predictions to probabilities
                        pred = predictions[name]
                        n_classes = len(np.unique(pred))
                        prob = np.eye(n_classes)[pred]
                        probabilities[name] = prob
                except Exception as e:
                    logger.warning(f"Probability calculation failed for model '{name}': {e}")

        if not probabilities:
            # Fallback to hard voting
            return self._hard_voting(predictions)

        # Weighted average of probabilities
        weighted_probs = None
        total_weight = 0

        for name, prob in probabilities.items():
            weight = self.weights.get(name, 1.0)
            if weighted_probs is None:
                weighted_probs = weight * prob
            else:
                weighted_probs += weight * prob
            total_weight += weight

        # Normalize
        if total_weight > 0:
            weighted_probs /= total_weight

        # Return class with highest probability
        return np.argmax(weighted_probs, axis=1)

    def predict_proba(self, X: Any):
        """Get ensemble prediction probabilities."""
        probabilities = {}

        # Get probabilities from all models
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)
                    probabilities[name] = prob
                elif hasattr(model, 'predict'):
                    # Convert predictions to probabilities
                    pred = model.predict(X)
                    n_classes = len(np.unique(pred))
                    prob = np.eye(n_classes)[pred]
                    probabilities[name] = prob
            except Exception as e:
                logger.warning(f"Probability calculation failed for model '{name}': {e}")

        if not probabilities:
            raise ValueError("No probability predictions available")

        # Weighted average
        weighted_probs = None
        total_weight = 0

        for name, prob in probabilities.items():
            weight = self.weights.get(name, 1.0)
            if weighted_probs is None:
                weighted_probs = weight * prob
            else:
                weighted_probs += weight * prob
            total_weight += weight

        # Normalize
        if total_weight > 0:
            weighted_probs /= total_weight

        return weighted_probs


class WeightedAverageEnsemble(BaseEnsemble):
    """Weighted average ensemble implementation."""

    def predict(self, X: Any):
        """Make predictions using weighted averaging."""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, X: Any):
        """Get weighted average probabilities."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")

        probabilities = {}

        # Get probabilities from all models
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)
                    probabilities[name] = prob
                elif hasattr(model, 'predict'):
                    # Convert predictions to probabilities
                    pred = model.predict(X)
                    if isinstance(pred, torch.Tensor):
                        pred = pred.cpu().numpy()
                    n_classes = len(np.unique(pred))
                    prob = np.eye(n_classes)[pred]
                    probabilities[name] = prob
                else:
                    logger.warning(f"Model '{name}' has no predict method")
            except Exception as e:
                logger.warning(f"Prediction failed for model '{name}': {e}")

        if not probabilities:
            raise ValueError("No valid predictions obtained")

        # Normalize weights
        self._normalize_weights()

        # Weighted average
        weighted_probs = None

        for name, prob in probabilities.items():
            weight = self.weights.get(name, 1.0)
            if weighted_probs is None:
                weighted_probs = weight * prob
            else:
                weighted_probs += weight * prob

        return weighted_probs


class StackingEnsemble(BaseEnsemble):
    """Stacking ensemble implementation."""

    def __init__(self, config: EnsembleConfig):
        super().__init__(config)
        self.meta_learner = None
        self._setup_meta_learner()

    def _setup_meta_learner(self):
        """Setup the meta-learner."""
        if self.config.meta_learner == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            self.meta_learner = LogisticRegression(random_state=42)
        elif self.config.meta_learner == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            self.meta_learner = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.config.meta_learner == "xgboost":
            try:
                import xgboost as xgb
                self.meta_learner = xgb.XGBClassifier(random_state=42)
            except ImportError:
                logger.warning("XGBoost not available, using LogisticRegression")
                from sklearn.linear_model import LogisticRegression
                self.meta_learner = LogisticRegression(random_state=42)
        else:
            raise ValueError(f"Unknown meta-learner: {self.config.meta_learner}")

    def fit(self, X: Any, y: Any):
        """Fit the stacking ensemble."""
        # Fit base models first
        super().fit(X, y)

        # Generate meta-features using cross-validation
        if self.config.use_cross_validation:
            meta_features = self._generate_cv_meta_features(X, y)
        else:
            meta_features = self._generate_meta_features(X)

        # Fit meta-learner
        self.meta_learner.fit(meta_features, y)

        logger.info("Stacking ensemble fitted successfully")
        return self

    def _generate_cv_meta_features(self, X: Any, y: Any):
        """Generate meta-features using cross-validation."""
        from sklearn.model_selection import cross_val_predict

        meta_features = []

        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    # Get cross-validated probabilities
                    cv_probs = cross_val_predict(
                        model, X, y, cv=self.config.cv_folds,
                        method='predict_proba'
                    )
                    meta_features.append(cv_probs)
                else:
                    # Get cross-validated predictions
                    cv_preds = cross_val_predict(
                        model, X, y, cv=self.config.cv_folds
                    )
                    # Convert to one-hot encoding
                    n_classes = len(np.unique(y))
                    cv_probs = np.eye(n_classes)[cv_preds]
                    meta_features.append(cv_probs)
            except Exception as e:
                logger.warning(f"CV meta-feature generation failed for '{name}': {e}")

        if not meta_features:
            raise ValueError("No meta-features generated")

        # Concatenate all meta-features
        return np.concatenate(meta_features, axis=1)

    def _generate_meta_features(self, X: Any):
        """Generate meta-features without cross-validation."""
        meta_features = []

        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)
                    meta_features.append(prob)
                else:
                    pred = model.predict(X)
                    n_classes = 2  # Assuming binary classification
                    prob = np.eye(n_classes)[pred]
                    meta_features.append(prob)
            except Exception as e:
                logger.warning(f"Meta-feature generation failed for '{name}': {e}")

        if not meta_features:
            raise ValueError("No meta-features generated")

        return np.concatenate(meta_features, axis=1)

    def predict(self, X: Any):
        """Make predictions using stacking."""
        meta_features = self._generate_meta_features(X)
        return self.meta_learner.predict(meta_features)

    def predict_proba(self, X: Any):
        """Get prediction probabilities using stacking."""
        meta_features = self._generate_meta_features(X)
        return self.meta_learner.predict_proba(meta_features)


class AdaptiveEnsemble(WeightedAverageEnsemble):
    """Adaptive ensemble that updates weights based on performance."""

    def __init__(self, config: EnsembleConfig):
        config.adaptive_weights = True
        super().__init__(config)
        self.performance_buffer = []

    def predict(self, X: Any):
        """Make adaptive predictions."""
        predictions = super().predict(X)

        # Store predictions for potential weight updates
        self.prediction_count += len(predictions)

        return predictions

    def update_weights(self, X: Any, y_true: Any):
        """Update weights based on recent performance."""
        # Get individual model predictions
        individual_predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                individual_predictions[name] = pred
            except Exception as e:
                logger.warning(f"Prediction failed for model '{name}': {e}")

        # Update weights
        self._update_adaptive_weights(individual_predictions, y_true)

        logger.info(f"Updated weights: {self.weights}")


# Factory functions
def create_ensemble(ensemble_type: str, config: EnsembleConfig = None):
    """Factory function to create ensemble."""
    if config is None:
        config = EnsembleConfig(ensemble_type=ensemble_type)

    if ensemble_type == "voting":
        return VotingEnsemble(config)
    elif ensemble_type == "weighted_average":
        return WeightedAverageEnsemble(config)
    elif ensemble_type == "stacking":
        return StackingEnsemble(config)
    elif ensemble_type == "adaptive":
        return AdaptiveEnsemble(config)
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


def evaluate_ensemble(ensemble, X_test: Any, y_test: Any):
    """Evaluate ensemble performance."""
    predictions = ensemble.predict(X_test)
    probabilities = ensemble.predict_proba(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'f1': f1_score(y_test, predictions, average='weighted'),
        'precision': precision_score(y_test, predictions, average='weighted'),
        'recall': recall_score(y_test, predictions, average='weighted')
    }

    # Add individual model performance if available
    if hasattr(ensemble, 'models'):
        individual_metrics = {}
        for name, model in ensemble.models.items():
            try:
                ind_pred = model.predict(X_test)
                individual_metrics[name] = {
                    'accuracy': accuracy_score(y_test, ind_pred),
                    'f1': f1_score(y_test, ind_pred, average='weighted')
                }
            except Exception as e:
                logger.warning(f"Individual evaluation failed for '{name}': {e}")

        metrics['individual_performance'] = individual_metrics

    return metrics


# Utility functions for model combination
def combine_tabular_image_models(tabular_model, image_model, weights: Dict[str, float] = None):
    """Combine tabular and image models into ensemble."""
    if weights is None:
        weights = {"tabular": 0.6, "image": 0.4}

    config = EnsembleConfig(
        ensemble_type="weighted_average",
        weights=weights
    )

    ensemble = create_ensemble("weighted_average", config)
    ensemble.add_model("tabular", tabular_model, weights["tabular"])
    ensemble.add_model("image", image_model, weights["image"])

    return ensemble


def optimize_ensemble_weights(models: Dict[str, Any], X_val: Any, y_val: Any):
    """Optimize ensemble weights using validation data."""
    from scipy.optimize import minimize

    model_names = list(models.keys())
    n_models = len(model_names)

    def objective(weights):
        # Create temporary ensemble
        config = EnsembleConfig(ensemble_type="weighted_average")
        ensemble = WeightedAverageEnsemble(config)

        for i, (name, model) in enumerate(models.items()):
            ensemble.add_model(name, model, weights[i])

        # Evaluate
        try:
            predictions = ensemble.predict(X_val)
            return -f1_score(y_val, predictions, average='weighted')
        except:
            return 1.0  # High penalty for failed predictions

    # Constraints: weights sum to 1, all positive
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0.01, 1.0) for _ in range(n_models)]

    # Initial weights (equal)
    initial_weights = np.ones(n_models) / n_models

    # Optimize
    result = minimize(
        objective, initial_weights,
        method='SLSQP', bounds=bounds,
        constraints=constraints
    )

    if result.success:
        optimal_weights = dict(zip(model_names, result.x))
        logger.info(f"Optimized weights: {optimal_weights}")
        return optimal_weights
    else:
        logger.warning("Weight optimization failed, using equal weights")
        return dict(zip(model_names, initial_weights))