"""
Cross-validation utilities for skin cancer detection models.

This module provides comprehensive cross-validation strategies including
stratified, time series, group-based, and custom validation schemes.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Iterator
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit,
    LeaveOneOut, LeavePOut, ShuffleSplit, StratifiedShuffleSplit,
    cross_val_score, cross_validate, validation_curve, learning_curve
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import clone
import warnings

logger = logging.getLogger(__name__)


@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation."""

    # CV strategy
    cv_strategy: str = "stratified_kfold"  # "kfold", "stratified_kfold", "group_kfold", etc.

    # Basic CV parameters
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42

    # Stratified parameters
    stratify_column: str = None

    # Group parameters
    group_column: str = None

    # Time series parameters
    max_train_size: int = None
    test_size: int = None
    gap: int = 0

    # Leave-P-Out parameters
    p: int = 1

    # Shuffle split parameters
    n_iter: int = 10
    test_ratio: float = 0.2

    # Scoring
    scoring: Union[str, List[str]] = "f1_weighted"

    # Parallel processing
    n_jobs: int = -1
    verbose: int = 0

    # Results
    return_train_score: bool = True
    return_estimator: bool = False

    # Visualization
    create_plots: bool = True
    plot_dir: str = "cv_plots"


class CrossValidator:
    """Comprehensive cross-validation utility."""

    def __init__(self, config: CrossValidationConfig = None):
        self.config = config or CrossValidationConfig()
        self.cv_results = {}
        self.cv_splitter = None

        # Setup plot directory
        from pathlib import Path
        self.plot_dir = Path(self.config.plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # Create CV splitter
        self._create_cv_splitter()

    def _create_cv_splitter(self):
        """Create cross-validation splitter based on configuration."""
        strategy = self.config.cv_strategy.lower()

        if strategy == "kfold":
            self.cv_splitter = KFold(
                n_splits=self.config.n_splits,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state
            )

        elif strategy == "stratified_kfold":
            self.cv_splitter = StratifiedKFold(
                n_splits=self.config.n_splits,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state
            )

        elif strategy == "group_kfold":
            self.cv_splitter = GroupKFold(n_splits=self.config.n_splits)

        elif strategy == "time_series_split":
            self.cv_splitter = TimeSeriesSplit(
                n_splits=self.config.n_splits,
                max_train_size=self.config.max_train_size,
                test_size=self.config.test_size,
                gap=self.config.gap
            )

        elif strategy == "leave_one_out":
            self.cv_splitter = LeaveOneOut()

        elif strategy == "leave_p_out":
            self.cv_splitter = LeavePOut(p=self.config.p)

        elif strategy == "shuffle_split":
            self.cv_splitter = ShuffleSplit(
                n_splits=self.config.n_iter,
                test_size=self.config.test_ratio,
                random_state=self.config.random_state
            )

        elif strategy == "stratified_shuffle_split":
            self.cv_splitter = StratifiedShuffleSplit(
                n_splits=self.config.n_iter,
                test_size=self.config.test_ratio,
                random_state=self.config.random_state
            )

        else:
            raise ValueError(f"Unknown CV strategy: {strategy}")

        logger.info(f"Created {strategy} cross-validator with {self.config.n_splits} splits")

    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series,
                           groups: pd.Series = None):
        """Perform cross-validation on a single model."""
        logger.info("Starting cross-validation...")

        # Prepare groups if needed
        if self.config.cv_strategy == "group_kfold":
            if groups is None and self.config.group_column:
                groups = X[self.config.group_column] if self.config.group_column in X.columns else None
            if groups is None:
                raise ValueError("Groups required for GroupKFold but not provided")

        # Perform cross-validation
        if isinstance(self.config.scoring, str):
            # Single scoring metric
            cv_scores = cross_val_score(
                model, X, y,
                cv=self.cv_splitter,
                scoring=self.config.scoring,
                groups=groups,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose
            )

            results = {
                'scores': cv_scores.tolist(),
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scoring': self.config.scoring
            }

        else:
            # Multiple scoring metrics
            cv_results = cross_validate(
                model, X, y,
                cv=self.cv_splitter,
                scoring=self.config.scoring,
                groups=groups,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose,
                return_train_score=self.config.return_train_score,
                return_estimator=self.config.return_estimator
            )

            results = self._format_cv_results(cv_results)

        # Store results
        model_name = getattr(model, '__class__', type(model)).__name__
        self.cv_results[model_name] = results

        logger.info(f"Cross-validation completed for {model_name}")
        return results

    def compare_models(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                      groups: pd.Series = None):
        """Compare multiple models using cross-validation."""
        logger.info(f"Comparing {len(models)} models using cross-validation")

        comparison_results = {}

        for model_name, model in models.items():
            logger.info(f"Cross-validating {model_name}...")

            try:
                results = self.cross_validate_model(model, X, y, groups)
                comparison_results[model_name] = results
            except Exception as e:
                logger.error(f"Cross-validation failed for {model_name}: {e}")
                comparison_results[model_name] = {'error': str(e)}

        # Create comparison visualizations
        if self.config.create_plots:
            self._create_comparison_plots(comparison_results)

        # Statistical comparison
        statistical_comparison = self._perform_statistical_comparison(comparison_results)

        final_results = {
            'individual_results': comparison_results,
            'statistical_comparison': statistical_comparison,
            'ranking': self._rank_models(comparison_results)
        }

        return final_results

    def nested_cross_validation(self, model, param_grid: Dict[str, List],
                               X: pd.DataFrame, y: pd.Series,
                               inner_cv_splits: int = 3):
        """Perform nested cross-validation for unbiased performance estimation."""
        from sklearn.model_selection import GridSearchCV

        logger.info("Starting nested cross-validation...")

        # Outer CV loop
        outer_scores = []
        best_params_list = []

        for fold, (train_idx, test_idx) in enumerate(self.cv_splitter.split(X, y)):
            logger.info(f"Outer fold {fold + 1}/{self.config.n_splits}")

            X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
            y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]

            # Inner CV loop (hyperparameter tuning)
            inner_cv = StratifiedKFold(
                n_splits=inner_cv_splits,
                shuffle=True,
                random_state=self.config.random_state
            )

            grid_search = GridSearchCV(
                model, param_grid,
                cv=inner_cv,
                scoring=self.config.scoring if isinstance(self.config.scoring, str) else self.config.scoring[0],
                n_jobs=self.config.n_jobs
            )

            # Fit on outer training set
            grid_search.fit(X_train_outer, y_train_outer)

            # Evaluate on outer test set
            best_model = grid_search.best_estimator_
            outer_score = best_model.score(X_test_outer, y_test_outer)

            outer_scores.append(outer_score)
            best_params_list.append(grid_search.best_params_)

        nested_results = {
            'outer_scores': outer_scores,
            'mean_score': np.mean(outer_scores),
            'std_score': np.std(outer_scores),
            'best_params_per_fold': best_params_list,
            'most_common_params': self._get_most_common_params(best_params_list)
        }

        logger.info(f"Nested CV completed. Mean score: {nested_results['mean_score']:.4f} ± {nested_results['std_score']:.4f}")

        return nested_results

    def learning_curve_analysis(self, model, X: pd.DataFrame, y: pd.Series,
                               train_sizes: np.ndarray = None):
        """Analyze learning curves using cross-validation."""
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        logger.info("Generating learning curves...")

        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=self.cv_splitter,
            scoring=self.config.scoring if isinstance(self.config.scoring, str) else self.config.scoring[0],
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose
        )

        # Calculate statistics
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Create plot
        if self.config.create_plots:
            self._plot_learning_curves(
                train_sizes_abs, train_mean, train_std, val_mean, val_std
            )

        learning_curve_results = {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores_mean': train_mean.tolist(),
            'train_scores_std': train_std.tolist(),
            'val_scores_mean': val_mean.tolist(),
            'val_scores_std': val_std.tolist()
        }

        return learning_curve_results

    def validation_curve_analysis(self, model, X: pd.DataFrame, y: pd.Series,
                                 param_name: str, param_range: List):
        """Analyze validation curves for a specific parameter."""
        logger.info(f"Generating validation curve for {param_name}...")

        train_scores, val_scores = validation_curve(
            model, X, y,
            param_name=param_name,
            param_range=param_range,
            cv=self.cv_splitter,
            scoring=self.config.scoring if isinstance(self.config.scoring, str) else self.config.scoring[0],
            n_jobs=self.config.n_jobs
        )

        # Calculate statistics
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Create plot
        if self.config.create_plots:
            self._plot_validation_curves(
                param_name, param_range, train_mean, train_std, val_mean, val_std
            )

        validation_curve_results = {
            'param_name': param_name,
            'param_range': param_range,
            'train_scores_mean': train_mean.tolist(),
            'train_scores_std': train_std.tolist(),
            'val_scores_mean': val_mean.tolist(),
            'val_scores_std': val_std.tolist(),
            'best_param_value': param_range[np.argmax(val_mean)]
        }

        return validation_curve_results

    def _format_cv_results(self, cv_results):
        """Format cross-validation results."""
        formatted_results = {}

        for key, values in cv_results.items():
            if key.startswith('test_'):
                metric = key.replace('test_', '')
                formatted_results[metric] = {
                    'test_scores': values.tolist(),
                    'test_mean': values.mean(),
                    'test_std': values.std()
                }

                # Add train scores if available
                train_key = f'train_{metric}'
                if train_key in cv_results:
                    train_values = cv_results[train_key]
                    formatted_results[metric].update({
                        'train_scores': train_values.tolist(),
                        'train_mean': train_values.mean(),
                        'train_std': train_values.std()
                    })

        return formatted_results

    def _create_comparison_plots(self, comparison_results):
        """Create comparison plots for multiple models."""
        valid_results = {k: v for k, v in comparison_results.items() if 'error' not in v}

        if not valid_results:
            logger.warning("No valid results for plotting")
            return

        # Extract metrics
        if isinstance(self.config.scoring, str):
            metric = self.config.scoring
            scores_data = {
                model: results['scores']
                for model, results in valid_results.items()
            }

            self._plot_cv_scores_comparison(scores_data, metric)

        else:
            # Multiple metrics
            for metric in self.config.scoring:
                scores_data = {
                    model: results[metric]['test_scores']
                    for model, results in valid_results.items()
                    if metric in results
                }

                if scores_data:
                    self._plot_cv_scores_comparison(scores_data, metric)

    def _plot_cv_scores_comparison(self, scores_data, metric):
        """Plot cross-validation scores comparison."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Box plot
        data_for_plot = []
        labels = []

        for model, scores in scores_data.items():
            data_for_plot.append(scores)
            labels.append(model)

        box_plot = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)

        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_title(f'Cross-Validation {metric.replace("_", " ").title()} Comparison')
        ax.set_ylabel(f'{metric.replace("_", " ").title()} Score')
        ax.set_xlabel('Models')
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        plot_path = self.plot_dir / f"cv_comparison_{metric}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"CV comparison plot saved: {plot_path}")

    def _plot_learning_curves(self, train_sizes, train_mean, train_std, val_mean, val_std):
        """Plot learning curves."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(train_sizes, train_mean, 'o-', label='Training Score')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)

        ax.plot(train_sizes, val_mean, 'o-', label='Validation Score')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)

        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Score')
        ax.set_title('Learning Curves')
        ax.legend()
        ax.grid(True)

        # Save plot
        plot_path = self.plot_dir / "learning_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_validation_curves(self, param_name, param_range, train_mean, train_std, val_mean, val_std):
        """Plot validation curves."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(param_range, train_mean, 'o-', label='Training Score')
        ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)

        ax.plot(param_range, val_mean, 'o-', label='Validation Score')
        ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1)

        ax.set_xlabel(param_name.replace('_', ' ').title())
        ax.set_ylabel('Score')
        ax.set_title(f'Validation Curve - {param_name}')
        ax.legend()
        ax.grid(True)

        # Save plot
        plot_path = self.plot_dir / f"validation_curve_{param_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _perform_statistical_comparison(self, comparison_results):
        """Perform statistical comparison between models."""
        from scipy import stats

        valid_results = {k: v for k, v in comparison_results.items() if 'error' not in v}
        model_names = list(valid_results.keys())

        if len(model_names) < 2:
            return None

        statistical_tests = {}

        # Get scores for each model
        if isinstance(self.config.scoring, str):
            model_scores = {
                name: results['scores']
                for name, results in valid_results.items()
            }

            # Pairwise comparisons
            for i, model1 in enumerate(model_names):
                for model2 in model_names[i+1:]:
                    scores1 = model_scores[model1]
                    scores2 = model_scores[model2]

                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(scores1, scores2)

                    # Wilcoxon signed-rank test
                    w_stat, w_p_value = stats.wilcoxon(scores1, scores2)

                    statistical_tests[f"{model1}_vs_{model2}"] = {
                        'paired_t_test': {
                            'statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        },
                        'wilcoxon_test': {
                            'statistic': float(w_stat),
                            'p_value': float(w_p_value),
                            'significant': w_p_value < 0.05
                        }
                    }

        return statistical_tests

    def _rank_models(self, comparison_results):
        """Rank models based on CV performance."""
        valid_results = {k: v for k, v in comparison_results.items() if 'error' not in v}

        if isinstance(self.config.scoring, str):
            # Single metric ranking
            model_scores = [
                (name, results['mean'])
                for name, results in valid_results.items()
            ]

            # Sort by score (descending)
            model_scores.sort(key=lambda x: x[1], reverse=True)

            return [
                {'rank': i+1, 'model': name, 'score': score}
                for i, (name, score) in enumerate(model_scores)
            ]

        else:
            # Multiple metrics - use first metric for ranking
            primary_metric = self.config.scoring[0]
            model_scores = [
                (name, results[primary_metric]['test_mean'])
                for name, results in valid_results.items()
                if primary_metric in results
            ]

            model_scores.sort(key=lambda x: x[1], reverse=True)

            return [
                {'rank': i+1, 'model': name, 'score': score}
                for i, (name, score) in enumerate(model_scores)
            ]

    def _get_most_common_params(self, params_list):
        """Get most common parameters from nested CV."""
        from collections import Counter

        # Flatten all parameters
        all_params = {}
        for params in params_list:
            for key, value in params.items():
                if key not in all_params:
                    all_params[key] = []
                all_params[key].append(value)

        # Find most common value for each parameter
        most_common = {}
        for key, values in all_params.items():
            counter = Counter(values)
            most_common[key] = counter.most_common(1)[0][0]

        return most_common


class TimeSeriesCrossValidator(CrossValidator):
    """Specialized cross-validator for time series data."""

    def __init__(self, config: CrossValidationConfig = None):
        if config is None:
            config = CrossValidationConfig(cv_strategy="time_series_split")
        super().__init__(config)

    def walk_forward_validation(self, model, X: pd.DataFrame, y: pd.Series,
                               initial_window: int = None, step_size: int = 1):
        """Perform walk-forward validation."""
        if initial_window is None:
            initial_window = len(X) // 3

        results = []

        for i in range(initial_window, len(X) - step_size + 1, step_size):
            # Training data
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]

            # Test data
            X_test = X.iloc[i:i+step_size]
            y_test = y.iloc[i:i+step_size]

            # Train and predict
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)

            # Calculate metrics
            score = f1_score(y_test, y_pred, average='weighted')
            results.append({
                'train_end': i,
                'test_start': i,
                'test_end': i + step_size,
                'score': score
            })

        return results


# Utility functions
def cross_validate_model(model, X, y, cv_strategy="stratified_kfold", **kwargs):
    """Quick cross-validation function."""
    config = CrossValidationConfig(cv_strategy=cv_strategy, **kwargs)
    cv = CrossValidator(config)
    return cv.cross_validate_model(model, X, y)


def compare_models_cv(models, X, y, **kwargs):
    """Quick model comparison using cross-validation."""
    config = CrossValidationConfig(**kwargs)
    cv = CrossValidator(config)
    return cv.compare_models(models, X, y)


def nested_cv(model, param_grid, X, y, **kwargs):
    """Quick nested cross-validation."""
    config = CrossValidationConfig(**kwargs)
    cv = CrossValidator(config)
    return cv.nested_cross_validation(model, param_grid, X, y)


# Custom CV splitters
class StratifiedGroupKFold:
    """Stratified Group K-Fold cross-validator."""

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y, groups):
        """Generate indices to split data into training and test set."""
        from sklearn.model_selection._split import _BaseKFold

        # Group by groups and get stratified splits
        unique_groups = np.unique(groups)

        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(unique_groups)

        # Split groups into folds
        fold_size = len(unique_groups) // self.n_splits

        for i in range(self.n_splits):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < self.n_splits - 1 else len(unique_groups)

            test_groups = unique_groups[start_idx:end_idx]
            train_groups = np.setdiff1d(unique_groups, test_groups)

            train_idx = np.where(np.isin(groups, train_groups))[0]
            test_idx = np.where(np.isin(groups, test_groups))[0]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations."""
        return self.n_splits