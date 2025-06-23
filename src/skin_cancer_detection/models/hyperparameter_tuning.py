"""
Hyperparameter tuning utilities for skin cancer detection models.

This module provides various hyperparameter optimization strategies including
grid search, random search, Bayesian optimization, and evolutionary algorithms.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold,
    cross_val_score, validation_curve
)
from sklearn.metrics import make_scorer, f1_score
import joblib
import json
from pathlib import Path

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterTuningConfig:
    """Configuration for hyperparameter tuning."""

    # Tuning strategy
    strategy: str = "random_search"  # "grid_search", "random_search", "bayesian", "optuna"

    # Search space
    param_space: Dict[str, Any] = field(default_factory=dict)

    # Cross-validation settings
    cv_folds: int = 5
    cv_scoring: str = "f1_weighted"
    cv_random_state: int = 42

    # Search settings
    n_iter: int = 100  # For random search and Bayesian optimization
    n_trials: int = 100  # For Optuna
    random_state: int = 42

    # Resource management
    n_jobs: int = -1
    verbose: int = 1

    # Early stopping
    early_stopping: bool = True
    patience: int = 10

    # Results management
    save_results: bool = True
    results_dir: str = "tuning_results"

    # Monitoring
    use_wandb: bool = False
    wandb_project: str = "hyperparameter-tuning"


class BaseHyperparameterTuner:
    """Base class for hyperparameter tuning."""

    def __init__(self, model_class, config: HyperparameterTuningConfig):
        self.model_class = model_class
        self.config = config
        self.best_params = None
        self.best_score = None
        self.best_model = None
        self.tuning_results = {}

        # Setup results directory
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize WandB if requested
        if config.use_wandb and WANDB_AVAILABLE:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        try:
            wandb.init(
                project=self.config.wandb_project,
                config=self.config.__dict__
            )
            logger.info("WandB initialized for hyperparameter tuning")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Perform hyperparameter tuning."""
        raise NotImplementedError("Subclasses must implement tune method")

    def get_best_model(self):
        """Get the best model found during tuning."""
        if self.best_model is None:
            raise ValueError("No tuning has been performed yet")
        return self.best_model

    def save_results(self, filename: str = None):
        """Save tuning results."""
        if filename is None:
            filename = f"{self.config.strategy}_results.json"

        filepath = self.results_dir / filename

        results_data = {
            'config': self.config.__dict__,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'tuning_results': self.tuning_results
        }

        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        logger.info(f"Tuning results saved to {filepath}")

    def load_results(self, filepath: str):
        """Load tuning results."""
        with open(filepath, 'r') as f:
            results_data = json.load(f)

        self.best_params = results_data['best_params']
        self.best_score = results_data['best_score']
        self.tuning_results = results_data['tuning_results']

        logger.info(f"Tuning results loaded from {filepath}")


class GridSearchTuner(BaseHyperparameterTuner):
    """Grid search hyperparameter tuner."""

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Perform grid search."""
        logger.info("Starting grid search hyperparameter tuning")

        # Setup cross-validation
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.cv_random_state
        )

        # Create base model
        base_model = self.model_class()

        # Setup grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.config.param_space,
            cv=cv,
            scoring=self.config.cv_scoring,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
            return_train_score=True
        )

        # Perform search
        grid_search.fit(X_train, y_train)

        # Store results
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.best_model = grid_search.best_estimator_

        # Store detailed results
        self.tuning_results = {
            'cv_results': grid_search.cv_results_,
            'total_combinations': len(grid_search.cv_results_['params']),
            'best_index': grid_search.best_index_
        }

        logger.info(f"Grid search completed. Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        return self.best_model


class RandomSearchTuner(BaseHyperparameterTuner):
    """Random search hyperparameter tuner."""

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Perform random search."""
        logger.info("Starting random search hyperparameter tuning")

        # Setup cross-validation
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.cv_random_state
        )

        # Create base model
        base_model = self.model_class()

        # Setup random search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=self.config.param_space,
            n_iter=self.config.n_iter,
            cv=cv,
            scoring=self.config.cv_scoring,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
            random_state=self.config.random_state,
            return_train_score=True
        )

        # Perform search
        random_search.fit(X_train, y_train)

        # Store results
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.best_model = random_search.best_estimator_

        # Store detailed results
        self.tuning_results = {
            'cv_results': random_search.cv_results_,
            'total_iterations': self.config.n_iter,
            'best_index': random_search.best_index_
        }

        logger.info(f"Random search completed. Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        return self.best_model


class BayesianTuner(BaseHyperparameterTuner):
    """Bayesian optimization hyperparameter tuner."""

    def __init__(self, model_class, config: HyperparameterTuningConfig):
        super().__init__(model_class, config)

        if not BAYESIAN_AVAILABLE:
            raise ImportError("scikit-optimize not available. Install with: pip install scikit-optimize")

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Perform Bayesian optimization."""
        logger.info("Starting Bayesian optimization hyperparameter tuning")

        # Setup cross-validation
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.cv_random_state
        )

        # Create base model
        base_model = self.model_class()

        # Setup Bayesian search
        bayes_search = BayesSearchCV(
            estimator=base_model,
            search_spaces=self.config.param_space,
            n_iter=self.config.n_iter,
            cv=cv,
            scoring=self.config.cv_scoring,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
            random_state=self.config.random_state,
            return_train_score=True
        )

        # Perform search
        bayes_search.fit(X_train, y_train)

        # Store results
        self.best_params = bayes_search.best_params_
        self.best_score = bayes_search.best_score_
        self.best_model = bayes_search.best_estimator_

        # Store detailed results
        self.tuning_results = {
            'cv_results': bayes_search.cv_results_,
            'total_iterations': self.config.n_iter,
            'best_index': bayes_search.best_index_,
            'optimizer_results': bayes_search.optimizer_results_
        }

        logger.info(f"Bayesian optimization completed. Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        return self.best_model


class OptunaTuner(BaseHyperparameterTuner):
    """Optuna hyperparameter tuner."""

    def __init__(self, model_class, config: HyperparameterTuningConfig):
        super().__init__(model_class, config)

        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")

        self.study = None

    def _objective(self, trial, X_train, y_train):
        """Objective function for Optuna."""
        # Suggest hyperparameters
        params = {}
        for param_name, param_config in self.config.param_space.items():
            if isinstance(param_config, dict):
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
            elif isinstance(param_config, list):
                params[param_name] = trial.suggest_categorical(param_name, param_config)

        # Create model with suggested parameters
        model = self.model_class(**params)

        # Cross-validation
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.cv_random_state
        )

        # Calculate cross-validation score
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring=self.config.cv_scoring
        )

        return scores.mean()

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Perform Optuna optimization."""
        logger.info("Starting Optuna hyperparameter tuning")

        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state)
        )

        # Add pruning if early stopping is enabled
        if self.config.early_stopping:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        else:
            pruner = optuna.pruners.NopPruner()

        # Optimize
        self.study.optimize(
            lambda trial: self._objective(trial, X_train, y_train),
            n_trials=self.config.n_trials,
            callbacks=[self._optuna_callback] if self.config.use_wandb else None
        )

        # Store results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        # Create best model
        self.best_model = self.model_class(**self.best_params)
        self.best_model.fit(X_train, y_train)

        # Store detailed results
        self.tuning_results = {
            'best_trial': self.study.best_trial._asdict(),
            'trials': [trial._asdict() for trial in self.study.trials],
            'total_trials': len(self.study.trials)
        }

        logger.info(f"Optuna optimization completed. Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        return self.best_model

    def _optuna_callback(self, study, trial):
        """Callback for logging to WandB."""
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'trial': trial.number,
                'value': trial.value,
                **trial.params
            })


class MultiModelTuner:
    """Tune hyperparameters for multiple models."""

    def __init__(self, models_config: Dict[str, Dict], config: HyperparameterTuningConfig):
        self.models_config = models_config
        self.config = config
        self.tuning_results = {}
        self.best_overall_model = None
        self.best_overall_score = -np.inf

    def tune_all_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Tune hyperparameters for all models."""
        logger.info(f"Starting hyperparameter tuning for {len(self.models_config)} models")

        for model_name, model_config in self.models_config.items():
            logger.info(f"Tuning {model_name}...")

            # Create tuning config for this model
            tuning_config = HyperparameterTuningConfig(
                strategy=self.config.strategy,
                param_space=model_config['param_space'],
                cv_folds=self.config.cv_folds,
                cv_scoring=self.config.cv_scoring,
                n_iter=self.config.n_iter,
                n_trials=self.config.n_trials,
                random_state=self.config.random_state,
                use_wandb=self.config.use_wandb,
                results_dir=f"{self.config.results_dir}/{model_name}"
            )

            # Create tuner
            tuner = create_tuner(
                model_config['model_class'],
                tuning_config
            )

            # Perform tuning
            try:
                best_model = tuner.tune(X_train, y_train)

                # Store results
                self.tuning_results[model_name] = {
                    'best_params': tuner.best_params,
                    'best_score': tuner.best_score,
                    'best_model': best_model,
                    'tuning_details': tuner.tuning_results
                }

                # Update overall best
                if tuner.best_score > self.best_overall_score:
                    self.best_overall_score = tuner.best_score
                    self.best_overall_model = best_model

                # Save individual results
                if self.config.save_results:
                    tuner.save_results()

                logger.info(f"{model_name} tuning completed. Best score: {tuner.best_score:.4f}")

            except Exception as e:
                logger.error(f"Tuning failed for {model_name}: {e}")
                self.tuning_results[model_name] = {'error': str(e)}

        logger.info("All model tuning completed")
        return self.tuning_results

    def get_best_model(self):
        """Get the overall best model."""
        return self.best_overall_model

    def get_comparison_results(self):
        """Get comparison of all tuned models."""
        comparison = {}

        for model_name, results in self.tuning_results.items():
            if 'error' not in results:
                comparison[model_name] = {
                    'best_score': results['best_score'],
                    'best_params': results['best_params']
                }

        # Sort by score
        sorted_comparison = dict(
            sorted(comparison.items(), key=lambda x: x[1]['best_score'], reverse=True)
        )

        return sorted_comparison


# Factory function
def create_tuner(model_class, config: HyperparameterTuningConfig):
    """Factory function to create hyperparameter tuner."""
    if config.strategy == "grid_search":
        return GridSearchTuner(model_class, config)
    elif config.strategy == "random_search":
        return RandomSearchTuner(model_class, config)
    elif config.strategy == "bayesian":
        return BayesianTuner(model_class, config)
    elif config.strategy == "optuna":
        return OptunaTuner(model_class, config)
    else:
        raise ValueError(f"Unknown tuning strategy: {config.strategy}")


# Predefined parameter spaces
def get_xgboost_param_space():
    """Get parameter space for XGBoost."""
    return {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0]
    }


def get_random_forest_param_space():
    """Get parameter space for Random Forest."""
    return {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }


def get_lightgbm_param_space():
    """Get parameter space for LightGBM."""
    return {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0],
        'num_leaves': [31, 50, 100, 150]
    }


def get_logistic_regression_param_space():
    """Get parameter space for Logistic Regression."""
    return {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 500, 1000, 2000]
    }


# Utility functions
def tune_model(model_class, X_train, y_train, strategy="random_search", **kwargs):
    """Quick model tuning function."""
    config = HyperparameterTuningConfig(strategy=strategy, **kwargs)
    tuner = create_tuner(model_class, config)
    return tuner.tune(X_train, y_train)


def tune_multiple_models(models_config, X_train, y_train, **kwargs):
    """Quick multiple model tuning function."""
    config = HyperparameterTuningConfig(**kwargs)
    multi_tuner = MultiModelTuner(models_config, config)
    return multi_tuner.tune_all_models(X_train, y_train)


def create_optuna_param_space(param_config):
    """Convert parameter configuration to Optuna format."""
    optuna_space = {}

    for param_name, param_def in param_config.items():
        if isinstance(param_def, list):
            # Categorical parameter
            optuna_space[param_name] = {
                'type': 'categorical',
                'choices': param_def
            }
        elif isinstance(param_def, dict):
            # Already in Optuna format
            optuna_space[param_name] = param_def
        elif isinstance(param_def, tuple) and len(param_def) == 2:
            # Range parameter
            if isinstance(param_def[0], int):
                optuna_space[param_name] = {
                    'type': 'int',
                    'low': param_def[0],
                    'high': param_def[1]
                }
            else:
                optuna_space[param_name] = {
                    'type': 'float',
                    'low': param_def[0],
                    'high': param_def[1]
                }

    return optuna_space