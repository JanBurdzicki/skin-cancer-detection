"""
Model evaluation utilities for skin cancer detection.

This module provides comprehensive evaluation tools including metrics calculation,
cross-validation, statistical tests, and performance visualization.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss
)
from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold,
    learning_curve, validation_curve
)
import scipy.stats as stats

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    # Cross-validation settings
    cv_folds: int = 5
    cv_scoring: Union[str, List[str]] = 'f1_weighted'
    cv_random_state: int = 42

    # Metrics to calculate
    metrics: List[str] = None

    # Visualization settings
    create_plots: bool = True
    plot_size: Tuple[int, int] = (10, 8)
    save_plots: bool = True
    plot_dir: str = "evaluation_plots"

    # Statistical testing
    perform_statistical_tests: bool = True
    significance_level: float = 0.05

    # Learning curve settings
    create_learning_curves: bool = True
    train_sizes: np.ndarray = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                'accuracy', 'precision', 'recall', 'f1',
                'roc_auc', 'matthews_corrcoef', 'cohen_kappa'
            ]

        if self.train_sizes is None:
            self.train_sizes = np.linspace(0.1, 1.0, 10)


class ModelEvaluator:
    """Comprehensive model evaluation utility."""

    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.results = {}
        self.comparison_results = {}

        # Setup plot directory
        from pathlib import Path
        self.plot_dir = Path(self.config.plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_single_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                             model_name: str = "model", X_train: pd.DataFrame = None,
                             y_train: pd.Series = None):
        """Evaluate a single model comprehensively."""
        logger.info(f"Evaluating model: {model_name}")

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None

        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)

        # Calculate metrics
        metrics = self._calculate_all_metrics(y_test, y_pred, y_pred_proba)

        # Cross-validation if training data is provided
        cv_results = None
        if X_train is not None and y_train is not None:
            cv_results = self._perform_cross_validation(model, X_train, y_train)

        # Create visualizations
        plots = {}
        if self.config.create_plots:
            plots = self._create_evaluation_plots(
                y_test, y_pred, y_pred_proba, model_name
            )

        # Learning curves
        learning_curve_data = None
        if self.config.create_learning_curves and X_train is not None:
            learning_curve_data = self._create_learning_curves(
                model, X_train, y_train, model_name
            )

        # Store results
        evaluation_result = {
            'model_name': model_name,
            'test_metrics': metrics,
            'cross_validation': cv_results,
            'plots': plots,
            'learning_curves': learning_curve_data,
            'test_size': len(y_test),
            'class_distribution': dict(pd.Series(y_test).value_counts()),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        self.results[model_name] = evaluation_result
        logger.info(f"Evaluation completed for {model_name}")

        return evaluation_result

    def compare_models(self, models: Dict[str, Any], X_test: pd.DataFrame,
                      y_test: pd.Series, X_train: pd.DataFrame = None,
                      y_train: pd.Series = None):
        """Compare multiple models."""
        logger.info(f"Comparing {len(models)} models")

        comparison_data = {}

        # Evaluate each model
        for model_name, model in models.items():
            result = self.evaluate_single_model(
                model, X_test, y_test, model_name, X_train, y_train
            )
            comparison_data[model_name] = result

        # Create comparison visualizations
        if self.config.create_plots:
            self._create_comparison_plots(comparison_data)

        # Statistical comparison
        statistical_comparison = None
        if self.config.perform_statistical_tests and len(models) > 1:
            statistical_comparison = self._perform_statistical_comparison(
                models, X_train, y_train
            )

        # Ranking
        ranking = self._rank_models(comparison_data)

        self.comparison_results = {
            'individual_results': comparison_data,
            'statistical_comparison': statistical_comparison,
            'ranking': ranking,
            'summary': self._create_comparison_summary(comparison_data)
        }

        return self.comparison_results

    def _calculate_all_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate all specified metrics."""
        metrics = {}

        # Basic classification metrics
        if 'accuracy' in self.config.metrics:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)

        if 'precision' in self.config.metrics:
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)

        if 'recall' in self.config.metrics:
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        if 'f1' in self.config.metrics:
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # ROC AUC (requires probabilities)
        if 'roc_auc' in self.config.metrics and y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except (ValueError, IndexError):
                metrics['roc_auc'] = 0.0

        # Average Precision Score
        if 'average_precision' in self.config.metrics and y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['average_precision'] = average_precision_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics['average_precision'] = average_precision_score(y_true, y_pred_proba, average='weighted')
            except (ValueError, IndexError):
                metrics['average_precision'] = 0.0

        # Matthews Correlation Coefficient
        if 'matthews_corrcoef' in self.config.metrics:
            metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)

        # Cohen's Kappa
        if 'cohen_kappa' in self.config.metrics:
            metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

        # Log Loss (requires probabilities)
        if 'log_loss' in self.config.metrics and y_pred_proba is not None:
            try:
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            except ValueError:
                metrics['log_loss'] = np.inf

        return metrics

    def _perform_cross_validation(self, model, X_train, y_train):
        """Perform cross-validation."""
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.cv_random_state
        )

        # Single scoring metric
        if isinstance(self.config.cv_scoring, str):
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, scoring=self.config.cv_scoring
            )

            return {
                'scores': cv_scores.tolist(),
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scoring': self.config.cv_scoring
            }

        # Multiple scoring metrics
        else:
            cv_results = cross_validate(
                model, X_train, y_train,
                cv=cv, scoring=self.config.cv_scoring,
                return_train_score=True
            )

            formatted_results = {}
            for metric in self.config.cv_scoring:
                test_scores = cv_results[f'test_{metric}']
                train_scores = cv_results[f'train_{metric}']

                formatted_results[metric] = {
                    'test_scores': test_scores.tolist(),
                    'train_scores': train_scores.tolist(),
                    'test_mean': test_scores.mean(),
                    'test_std': test_scores.std(),
                    'train_mean': train_scores.mean(),
                    'train_std': train_scores.std()
                }

            return formatted_results

    def _create_evaluation_plots(self, y_true, y_pred, y_pred_proba, model_name):
        """Create evaluation plots."""
        plots = {}

        # Confusion Matrix
        cm_fig = self._plot_confusion_matrix(y_true, y_pred, model_name)
        cm_path = self.plot_dir / f"{model_name}_confusion_matrix.png"
        if self.config.save_plots:
            cm_fig.savefig(cm_path, dpi=300, bbox_inches='tight')
        plots['confusion_matrix'] = str(cm_path)
        plt.close(cm_fig)

        # ROC Curve
        if y_pred_proba is not None:
            roc_fig = self._plot_roc_curve(y_true, y_pred_proba, model_name)
            roc_path = self.plot_dir / f"{model_name}_roc_curve.png"
            if self.config.save_plots:
                roc_fig.savefig(roc_path, dpi=300, bbox_inches='tight')
            plots['roc_curve'] = str(roc_path)
            plt.close(roc_fig)

            # Precision-Recall Curve
            pr_fig = self._plot_precision_recall_curve(y_true, y_pred_proba, model_name)
            pr_path = self.plot_dir / f"{model_name}_pr_curve.png"
            if self.config.save_plots:
                pr_fig.savefig(pr_path, dpi=300, bbox_inches='tight')
            plots['pr_curve'] = str(pr_path)
            plt.close(pr_fig)

        return plots

    def _plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Create confusion matrix plot."""
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=self.config.plot_size)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{model_name} - Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        return fig

    def _plot_roc_curve(self, y_true, y_pred_proba, model_name):
        """Create ROC curve plot."""
        fig, ax = plt.subplots(figsize=self.config.plot_size)

        if len(np.unique(y_true)) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])

            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{model_name} - ROC Curve')
            ax.legend()
        else:
            # Multi-class
            for i in range(y_pred_proba.shape[1]):
                fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
                auc_score = roc_auc_score(y_true == i, y_pred_proba[:, i])
                ax.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.3f})')

            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{model_name} - ROC Curves')
            ax.legend()

        return fig

    def _plot_precision_recall_curve(self, y_true, y_pred_proba, model_name):
        """Create precision-recall curve plot."""
        fig, ax = plt.subplots(figsize=self.config.plot_size)

        if len(np.unique(y_true)) == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
            ap_score = average_precision_score(y_true, y_pred_proba[:, 1])

            ax.plot(recall, precision, label=f'PR Curve (AP = {ap_score:.3f})')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'{model_name} - Precision-Recall Curve')
            ax.legend()

        return fig

    def _create_learning_curves(self, model, X_train, y_train, model_name):
        """Create learning curves."""
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train,
                train_sizes=self.config.train_sizes,
                cv=self.config.cv_folds,
                scoring='f1_weighted',
                random_state=self.config.cv_random_state
            )

            # Create plot
            fig, ax = plt.subplots(figsize=self.config.plot_size)

            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)

            ax.plot(train_sizes, train_mean, label='Training Score', marker='o')
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)

            ax.plot(train_sizes, val_mean, label='Validation Score', marker='s')
            ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)

            ax.set_xlabel('Training Set Size')
            ax.set_ylabel('F1 Score')
            ax.set_title(f'{model_name} - Learning Curves')
            ax.legend()
            ax.grid(True)

            # Save plot
            if self.config.save_plots:
                lc_path = self.plot_dir / f"{model_name}_learning_curves.png"
                fig.savefig(lc_path, dpi=300, bbox_inches='tight')

            plt.close(fig)

            return {
                'train_sizes': train_sizes.tolist(),
                'train_scores_mean': train_mean.tolist(),
                'train_scores_std': train_std.tolist(),
                'val_scores_mean': val_mean.tolist(),
                'val_scores_std': val_std.tolist()
            }

        except Exception as e:
            logger.warning(f"Learning curve creation failed for {model_name}: {e}")
            return None

    def _create_comparison_plots(self, comparison_data):
        """Create model comparison plots."""
        # Extract metrics for comparison
        models = list(comparison_data.keys())
        metrics_data = {}

        for metric in self.config.metrics:
            metrics_data[metric] = []
            for model_name in models:
                metric_value = comparison_data[model_name]['test_metrics'].get(metric, 0)
                metrics_data[metric].append(metric_value)

        # Create comparison bar chart
        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(models))
        width = 0.8 / len(self.config.metrics)

        for i, metric in enumerate(self.config.metrics):
            if metric in metrics_data:
                ax.bar(x + i * width, metrics_data[metric], width,
                      label=metric.replace('_', ' ').title(), alpha=0.8)

        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * (len(self.config.metrics) - 1) / 2)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if self.config.save_plots:
            comp_path = self.plot_dir / "model_comparison.png"
            fig.savefig(comp_path, dpi=300, bbox_inches='tight')

        plt.close(fig)

    def _perform_statistical_comparison(self, models, X_train, y_train):
        """Perform statistical comparison between models."""
        if len(models) < 2:
            return None

        model_names = list(models.keys())
        cv_results = {}

        # Get cross-validation scores for each model
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.cv_random_state
        )

        for name, model in models.items():
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
            cv_results[name] = scores

        # Perform pairwise statistical tests
        statistical_tests = {}

        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(cv_results[model1], cv_results[model2])

                # Wilcoxon signed-rank test (non-parametric alternative)
                w_stat, w_p_value = stats.wilcoxon(cv_results[model1], cv_results[model2])

                statistical_tests[f"{model1}_vs_{model2}"] = {
                    'paired_t_test': {
                        'statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < self.config.significance_level
                    },
                    'wilcoxon_test': {
                        'statistic': float(w_stat),
                        'p_value': float(w_p_value),
                        'significant': w_p_value < self.config.significance_level
                    },
                    'mean_difference': float(np.mean(cv_results[model1]) - np.mean(cv_results[model2]))
                }

        return {
            'cv_scores': {name: scores.tolist() for name, scores in cv_results.items()},
            'statistical_tests': statistical_tests
        }

    def _rank_models(self, comparison_data):
        """Rank models based on performance."""
        models = list(comparison_data.keys())

        # Create ranking for each metric
        metric_rankings = {}
        for metric in self.config.metrics:
            metric_scores = []
            for model_name in models:
                score = comparison_data[model_name]['test_metrics'].get(metric, 0)
                metric_scores.append((model_name, score))

            # Sort by score (descending for most metrics, ascending for loss metrics)
            reverse = metric not in ['log_loss']
            metric_scores.sort(key=lambda x: x[1], reverse=reverse)

            metric_rankings[metric] = [
                {'rank': i+1, 'model': name, 'score': score}
                for i, (name, score) in enumerate(metric_scores)
            ]

        # Overall ranking (average rank across metrics)
        overall_ranks = {}
        for model in models:
            ranks = []
            for metric in self.config.metrics:
                for entry in metric_rankings[metric]:
                    if entry['model'] == model:
                        ranks.append(entry['rank'])
                        break
            overall_ranks[model] = np.mean(ranks) if ranks else len(models)

        # Sort by average rank
        overall_ranking = sorted(overall_ranks.items(), key=lambda x: x[1])

        return {
            'metric_rankings': metric_rankings,
            'overall_ranking': [
                {'rank': i+1, 'model': name, 'average_rank': avg_rank}
                for i, (name, avg_rank) in enumerate(overall_ranking)
            ]
        }

    def _create_comparison_summary(self, comparison_data):
        """Create summary of model comparison."""
        models = list(comparison_data.keys())

        summary = {
            'total_models': len(models),
            'best_models': {},
            'metric_summary': {}
        }

        # Find best model for each metric
        for metric in self.config.metrics:
            best_score = -np.inf if metric != 'log_loss' else np.inf
            best_model = None

            for model_name in models:
                score = comparison_data[model_name]['test_metrics'].get(metric, 0)

                if metric == 'log_loss':
                    if score < best_score:
                        best_score = score
                        best_model = model_name
                else:
                    if score > best_score:
                        best_score = score
                        best_model = model_name

            summary['best_models'][metric] = {
                'model': best_model,
                'score': best_score
            }

        # Metric summary statistics
        for metric in self.config.metrics:
            scores = [
                comparison_data[model]['test_metrics'].get(metric, 0)
                for model in models
            ]

            summary['metric_summary'][metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }

        return summary

    def generate_evaluation_report(self, output_path: str = "evaluation_report.json"):
        """Generate comprehensive evaluation report."""
        import json
        from pathlib import Path

        report = {
            'config': self.config.__dict__,
            'individual_results': self.results,
            'comparison_results': self.comparison_results,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Save report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Evaluation report saved to {output_file}")
        return report


# Utility functions
def evaluate_model(model, X_test, y_test, model_name="model", **kwargs):
    """Quick model evaluation function."""
    config = EvaluationConfig(**kwargs)
    evaluator = ModelEvaluator(config)
    return evaluator.evaluate_single_model(model, X_test, y_test, model_name)


def compare_models(models, X_test, y_test, X_train=None, y_train=None, **kwargs):
    """Quick model comparison function."""
    config = EvaluationConfig(**kwargs)
    evaluator = ModelEvaluator(config)
    return evaluator.compare_models(models, X_test, y_test, X_train, y_train)


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate standard classification metrics."""
    config = EvaluationConfig()
    evaluator = ModelEvaluator(config)
    return evaluator._calculate_all_metrics(y_true, y_pred, y_pred_proba)