"""
Comprehensive model trainer utility for skin cancer detection.

This module provides unified training functionality for both image and tabular
models with MLflow and Wandb integration, hyperparameter optimization, and
comprehensive evaluation.
"""

import os
import logging
import mlflow
import wandb
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import optuna
from optuna.integration import MLflowCallback
import json

logger = logging.getLogger(__name__)


class BaseModelTrainer:
    """Base class for model training with common functionality."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.best_model = None
        self.training_history = {}
        self.evaluation_metrics = {}

        # Setup experiment tracking
        self.setup_experiment_tracking()

    def setup_experiment_tracking(self):
        """Setup MLflow and Wandb experiment tracking."""
        # MLflow setup
        mlflow.set_experiment(self.config.get('experiment_name', 'skin_cancer_detection'))

        # Wandb setup
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'skin-cancer-detection'),
                name=self.config.get('run_name'),
                config=self.config,
                tags=self.config.get('tags', [])
            )

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow and Wandb."""
        # Log to MLflow
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value, step=step)

        # Log to Wandb
        if self.config.get('use_wandb', False):
            wandb.log(metrics, step=step)

    def save_model(self, model_path: str, metadata: Dict[str, Any] = None):
        """Save model with metadata."""
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        if hasattr(self.model, 'state_dict'):  # PyTorch model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'metadata': metadata or {}
            }, model_path)
        else:  # Sklearn model
            joblib.dump({
                'model': self.model,
                'config': self.config,
                'metadata': metadata or {}
            }, model_path)

        # Log model artifact
        mlflow.log_artifact(model_path)
        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        """Load model from path."""
        if model_path.endswith('.pth') or model_path.endswith('.pt'):
            checkpoint = torch.load(model_path)
            self.config = checkpoint['config']
            # Model loading will be handled by specific trainer classes
        else:
            checkpoint = joblib.load(model_path)
            self.model = checkpoint['model']
            self.config = checkpoint['config']

        logger.info(f"Model loaded from {model_path}")
        return checkpoint.get('metadata', {})


class ImageModelTrainer(BaseModelTrainer):
    """Trainer for image classification models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def create_model(self) -> nn.Module:
        """Create image classification model."""
        from ..models.image_models import create_image_model

        model = create_image_model(
            model_type=self.config['model_type'],
            num_classes=self.config['num_classes'],
            pretrained=self.config.get('pretrained', True)
        )
        return model.to(self.device)

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              test_loader: Optional[DataLoader] = None) -> nn.Module:
        """Train image model."""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.config)

            # Create model
            self.model = self.create_model()

            # Setup optimizer and scheduler
            optimizer = self._create_optimizer()
            scheduler = self._create_scheduler(optimizer)
            criterion = self._create_criterion()

            # Training loop
            best_val_acc = 0.0
            for epoch in range(self.config['epochs']):
                # Train epoch
                train_metrics = self._train_epoch(train_loader, optimizer, criterion, epoch)

                # Validate epoch
                val_metrics = self._validate_epoch(val_loader, criterion, epoch)

                # Update scheduler
                if scheduler:
                    scheduler.step(val_metrics['val_loss'])

                # Save best model
                if val_metrics['val_accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['val_accuracy']
                    self.best_model = self.model.state_dict().copy()

                # Log metrics
                metrics = {**train_metrics, **val_metrics}
                self.log_metrics(metrics, step=epoch)

                logger.info(f"Epoch {epoch+1}/{self.config['epochs']}: "
                          f"Train Loss: {train_metrics['train_loss']:.4f}, "
                          f"Val Loss: {val_metrics['val_loss']:.4f}, "
                          f"Val Acc: {val_metrics['val_accuracy']:.4f}")

            # Load best model
            self.model.load_state_dict(self.best_model)

            # Final evaluation
            if test_loader:
                test_metrics = self.evaluate(test_loader)
                self.log_metrics(test_metrics)
                logger.info(f"Test metrics: {test_metrics}")

            return self.model

    def _train_epoch(self, train_loader: DataLoader, optimizer, criterion, epoch: int) -> Dict[str, float]:
        """Train single epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        return {
            'train_loss': running_loss / len(train_loader),
            'train_accuracy': correct / total
        }

    def _validate_epoch(self, val_loader: DataLoader, criterion, epoch: int) -> Dict[str, float]:
        """Validate single epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)

                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return {
            'val_loss': running_loss / len(val_loader),
            'val_accuracy': correct / total
        }

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Calculate metrics
        metrics = self._calculate_metrics(all_targets, all_predictions, all_probabilities)
        return metrics

    def _calculate_metrics(self, y_true: List, y_pred: List, y_prob: List) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }

        # Add ROC AUC for binary classification
        if len(np.unique(y_true)) == 2:
            y_prob_binary = [prob[1] for prob in y_prob]  # Probability of positive class
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob_binary)

        return metrics

    def _create_optimizer(self):
        """Create optimizer."""
        optimizer_type = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-4)

        if optimizer_type == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    def _create_scheduler(self, optimizer):
        """Create learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', 'plateau').lower()

        if scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        elif scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'])
        elif scheduler_type == 'step':
            step_size = self.config.get('step_size', 30)
            gamma = self.config.get('gamma', 0.1)
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            return None

    def _create_criterion(self):
        """Create loss criterion."""
        criterion_type = self.config.get('criterion', 'crossentropy').lower()

        if criterion_type == 'crossentropy':
            return nn.CrossEntropyLoss()
        elif criterion_type == 'focal':
            from ..models.losses import FocalLoss
            alpha = self.config.get('focal_alpha', 1.0)
            gamma = self.config.get('focal_gamma', 2.0)
            return FocalLoss(alpha=alpha, gamma=gamma)
        else:
            raise ValueError(f"Unsupported criterion: {criterion_type}")


class TabularModelTrainer(BaseModelTrainer):
    """Trainer for tabular classification models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def create_model(self):
        """Create tabular classification model."""
        from ..models.tabular_models import create_tabular_model

        model = create_tabular_model(
            model_type=self.config['model_type'],
            **self.config.get('model_params', {})
        )
        return model

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series,
              X_test: Optional[pd.DataFrame] = None, y_test: Optional[pd.Series] = None):
        """Train tabular model."""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.config)

            # Create model
            self.model = self.create_model()

            # Train model
            if hasattr(self.model, 'fit'):
                if self.config['model_type'] in ['xgboost', 'lightgbm', 'catboost']:
                    # Models that support validation data
                    if hasattr(self.model, 'fit') and 'eval_set' in self.model.fit.__code__.co_varnames:
                        self.model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=self.config.get('verbose', False)
                        )
                    else:
                        self.model.fit(X_train, y_train)
                else:
                    self.model.fit(X_train, y_train)

            # Validation evaluation
            val_predictions = self.model.predict(X_val)
            val_probabilities = self.model.predict_proba(X_val) if hasattr(self.model, 'predict_proba') else None
            val_metrics = self._calculate_metrics(y_val, val_predictions, val_probabilities)
            val_metrics = {f'val_{k}': v for k, v in val_metrics.items()}

            # Test evaluation
            if X_test is not None and y_test is not None:
                test_predictions = self.model.predict(X_test)
                test_probabilities = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
                test_metrics = self._calculate_metrics(y_test, test_predictions, test_probabilities)
                test_metrics = {f'test_{k}': v for k, v in test_metrics.items()}

                # Combine metrics
                all_metrics = {**val_metrics, **test_metrics}
            else:
                all_metrics = val_metrics

            # Log metrics
            self.log_metrics(all_metrics)
            logger.info(f"Training completed. Validation metrics: {val_metrics}")

            return self.model

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None

        metrics = self._calculate_metrics(y_test, predictions, probabilities)
        return metrics

    def _calculate_metrics(self, y_true, y_pred, y_prob=None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }

        # Add ROC AUC for binary classification
        if len(np.unique(y_true)) == 2 and y_prob is not None:
            y_prob_binary = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob_binary)

        return metrics


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.study = None

    def optimize_image_model(self, train_loader: DataLoader, val_loader: DataLoader,
                           n_trials: int = 50) -> Dict[str, Any]:
        """Optimize hyperparameters for image model."""

        def objective(trial):
            # Define hyperparameter search space
            config = self.config.copy()
            config.update({
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
                'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),
                'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'step', 'plateau']),
                'epochs': 10  # Reduced for optimization
            })

            # Train model
            trainer = ImageModelTrainer(config)
            trainer.train(train_loader, val_loader)

            # Evaluate on validation set
            val_metrics = trainer.evaluate(val_loader)

            return val_metrics['accuracy']

        # Create study
        mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri())
        self.study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.TPESampler()
        )

        # Optimize
        self.study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_callback])

        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'study': self.study
        }

    def optimize_tabular_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series,
                             n_trials: int = 50) -> Dict[str, Any]:
        """Optimize hyperparameters for tabular model."""

        def objective(trial):
            # Define hyperparameter search space based on model type
            config = self.config.copy()
            model_type = config['model_type']

            if model_type == 'xgboost':
                config['model_params'] = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
                }
            elif model_type == 'lightgbm':
                config['model_params'] = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
                }
            elif model_type == 'random_forest':
                config['model_params'] = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }

            # Train model
            trainer = TabularModelTrainer(config)
            trainer.train(X_train, y_train, X_val, y_val)

            # Evaluate on validation set
            val_metrics = trainer.evaluate(X_val, y_val)

            return val_metrics['accuracy']

        # Create study
        mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri())
        self.study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.TPESampler()
        )

        # Optimize
        self.study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_callback])

        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'study': self.study
        }


class CrossValidator:
    """Cross-validation for model evaluation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cv_results = {}

    def cross_validate_tabular(self, X: pd.DataFrame, y: pd.Series,
                             n_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation for tabular model."""
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Training fold {fold + 1}/{n_folds}")

            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Train model
            trainer = TabularModelTrainer(self.config)
            trainer.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold)

            # Evaluate
            metrics = trainer.evaluate(X_val_fold, y_val_fold)

            for metric, value in metrics.items():
                if metric in fold_scores:
                    fold_scores[metric].append(value)

        # Calculate mean and std
        cv_results = {}
        for metric, scores in fold_scores.items():
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)

        self.cv_results = cv_results
        return cv_results