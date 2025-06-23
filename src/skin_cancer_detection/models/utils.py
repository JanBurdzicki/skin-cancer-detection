"""
Utility functions for skin cancer detection models.

This module provides various utility functions for model operations including
data preprocessing, model evaluation, visualization, and file operations.
"""

import logging
import os
import pickle
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import joblib

logger = logging.getLogger(__name__)


# Data preprocessing utilities
def preprocess_tabular_data(df: pd.DataFrame, target_column: str = None,
                           test_size: float = 0.2, random_state: int = 42):
    """Preprocess tabular data for modeling."""
    df_processed = df.copy()

    # Handle missing values
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    categorical_columns = df_processed.select_dtypes(include=['object']).columns

    # Fill numeric missing values with median
    for col in numeric_columns:
        if df_processed[col].isnull().any():
            df_processed[col].fillna(df_processed[col].median(), inplace=True)

    # Fill categorical missing values with mode
    for col in categorical_columns:
        if col != target_column and df_processed[col].isnull().any():
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for col in categorical_columns:
        if col != target_column:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le

    # Separate features and target
    if target_column:
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]

        # Encode target if categorical
        if y.dtype == 'object':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
            label_encoders[target_column] = target_encoder

        return X, y, label_encoders
    else:
        return df_processed, label_encoders


def create_feature_subsets(df: pd.DataFrame, feature_groups: Dict[str, List[str]] = None):
    """Create different feature subsets for experimentation."""
    if feature_groups is None:
        # Default feature groups
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()

        feature_groups = {
            'all_features': df.columns.tolist(),
            'numeric_only': numeric_features,
            'categorical_only': categorical_features,
            'top_10': df.columns.tolist()[:10] if len(df.columns) >= 10 else df.columns.tolist()
        }

    feature_subsets = {}
    for group_name, features in feature_groups.items():
        available_features = [f for f in features if f in df.columns]
        if available_features:
            feature_subsets[group_name] = df[available_features]

    return feature_subsets


def balance_dataset(X: pd.DataFrame, y: pd.Series, method: str = "smote"):
    """Balance dataset using various techniques."""
    try:
        if method == "smote":
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)

        elif method == "undersample":
            from imblearn.under_sampling import RandomUnderSampler
            undersampler = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = undersampler.fit_resample(X, y)
            return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)

        elif method == "oversample":
            from imblearn.over_sampling import RandomOverSampler
            oversampler = RandomOverSampler(random_state=42)
            X_balanced, y_balanced = oversampler.fit_resample(X, y)
            return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)

        else:
            logger.warning(f"Unknown balancing method: {method}")
            return X, y

    except ImportError:
        logger.warning("imbalanced-learn not available, skipping dataset balancing")
        return X, y


# Model evaluation utilities
def calculate_comprehensive_metrics(y_true, y_pred, y_prob=None, average='weighted'):
    """Calculate comprehensive evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }

    # Add ROC AUC if probabilities are provided
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
            else:  # Multi-class
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)
        except (ValueError, IndexError):
            metrics['roc_auc'] = 0.0

    return metrics


def create_confusion_matrix_plot(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """Create confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    return plt.gcf()


def create_roc_curve_plot(y_true, y_prob, title="ROC Curve"):
    """Create ROC curve visualization."""
    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
        # Multi-class or binary with probabilities for both classes
        if len(np.unique(y_true)) == 2:
            y_prob_positive = y_prob[:, 1]
        else:
            # For multi-class, plot ROC for each class
            plt.figure(figsize=(10, 8))
            for i in range(y_prob.shape[1]):
                fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
                auc = roc_auc_score(y_true == i, y_prob[:, i])
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            return plt.gcf()
    else:
        y_prob_positive = y_prob

    # Binary classification ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob_positive)
    auc = roc_auc_score(y_true, y_prob_positive)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    return plt.gcf()


def create_precision_recall_curve_plot(y_true, y_prob, title="Precision-Recall Curve"):
    """Create precision-recall curve visualization."""
    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
        y_prob_positive = y_prob[:, 1]
    else:
        y_prob_positive = y_prob

    precision, recall, _ = precision_recall_curve(y_true, y_prob_positive)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    return plt.gcf()


def create_feature_importance_plot(importance_dict: Dict[str, float], top_n: int = 15,
                                 title="Feature Importance"):
    """Create feature importance visualization."""
    # Sort features by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_features[:top_n]

    features, importance = zip(*top_features)

    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    bars = plt.barh(range(len(features)), importance, color=colors)

    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()

    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars, importance)):
        plt.text(bar.get_width() + 0.01 * max(importance),
                bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', va='center', ha='left')

    plt.tight_layout()
    return plt.gcf()


def compare_models_performance(results: Dict[str, Dict[str, float]],
                             metrics: List[str] = None, title="Model Comparison"):
    """Create model performance comparison visualization."""
    if metrics is None:
        metrics = ['accuracy', 'f1', 'precision', 'recall']

    # Prepare data for plotting
    models = list(results.keys())
    metric_values = {metric: [] for metric in metrics}

    for model in models:
        for metric in metrics:
            metric_values[metric].append(results[model].get(metric, 0))

    # Create grouped bar chart
    x = np.arange(len(models))
    width = 0.8 / len(metrics)

    plt.figure(figsize=(12, 8))

    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, metric_values[metric], width,
                label=metric.capitalize(), alpha=0.8)

    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks(x + width * (len(metrics) - 1) / 2, models, rotation=45)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()

    return plt.gcf()


# File and model management utilities
def save_model_artifacts(model, model_name: str, output_dir: str = "models",
                        metadata: Dict[str, Any] = None):
    """Save model and associated artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    artifacts = {}

    # Save model
    if hasattr(model, 'save_model'):
        # XGBoost, LightGBM models
        model_path = output_path / f"{model_name}.model"
        model.save_model(str(model_path))
        artifacts['model_path'] = str(model_path)
    elif hasattr(model, 'state_dict'):
        # PyTorch models
        model_path = output_path / f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)
        artifacts['model_path'] = str(model_path)
    else:
        # Scikit-learn and other models
        model_path = output_path / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        artifacts['model_path'] = str(model_path)

    # Save metadata
    if metadata:
        metadata_path = output_path / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        artifacts['metadata_path'] = str(metadata_path)

    logger.info(f"Model artifacts saved to {output_path}")
    return artifacts


def load_model_artifacts(model_name: str, model_dir: str = "models"):
    """Load model and associated artifacts."""
    model_path = Path(model_dir)

    # Try different file extensions
    model_file = None
    for ext in ['.pkl', '.pth', '.model']:
        candidate = model_path / f"{model_name}{ext}"
        if candidate.exists():
            model_file = candidate
            break

    if model_file is None:
        raise FileNotFoundError(f"Model file not found for {model_name}")

    # Load model
    if model_file.suffix == '.pkl':
        model = joblib.load(model_file)
    elif model_file.suffix == '.pth':
        # For PyTorch models, you need to provide the model architecture
        model = torch.load(model_file, map_location='cpu')
    else:
        raise ValueError(f"Unsupported model file format: {model_file.suffix}")

    # Load metadata if available
    metadata_file = model_path / f"{model_name}_metadata.json"
    metadata = None
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

    return model, metadata


def create_model_report(model, X_test: pd.DataFrame, y_test: pd.Series,
                       model_name: str, output_dir: str = "reports"):
    """Create comprehensive model evaluation report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)

    # Calculate metrics
    metrics = calculate_comprehensive_metrics(y_test, y_pred, y_prob)

    # Create visualizations
    plots = {}

    # Confusion matrix
    cm_fig = create_confusion_matrix_plot(y_test, y_pred,
                                        title=f"{model_name} - Confusion Matrix")
    cm_path = output_path / f"{model_name}_confusion_matrix.png"
    cm_fig.savefig(cm_path, dpi=300, bbox_inches='tight')
    plots['confusion_matrix'] = str(cm_path)
    plt.close(cm_fig)

    # ROC curve (if probabilities available)
    if y_prob is not None:
        roc_fig = create_roc_curve_plot(y_test, y_prob,
                                      title=f"{model_name} - ROC Curve")
        roc_path = output_path / f"{model_name}_roc_curve.png"
        roc_fig.savefig(roc_path, dpi=300, bbox_inches='tight')
        plots['roc_curve'] = str(roc_path)
        plt.close(roc_fig)

        # Precision-Recall curve
        pr_fig = create_precision_recall_curve_plot(y_test, y_prob,
                                                  title=f"{model_name} - Precision-Recall Curve")
        pr_path = output_path / f"{model_name}_pr_curve.png"
        pr_fig.savefig(pr_path, dpi=300, bbox_inches='tight')
        plots['pr_curve'] = str(pr_path)
        plt.close(pr_fig)

    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        importance_dict = dict(zip(X_test.columns, model.feature_importances_))
        fi_fig = create_feature_importance_plot(importance_dict,
                                              title=f"{model_name} - Feature Importance")
        fi_path = output_path / f"{model_name}_feature_importance.png"
        fi_fig.savefig(fi_path, dpi=300, bbox_inches='tight')
        plots['feature_importance'] = str(fi_path)
        plt.close(fi_fig)

    # Create report
    report = {
        'model_name': model_name,
        'metrics': metrics,
        'plots': plots,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'test_size': len(y_test),
        'class_distribution': dict(pd.Series(y_test).value_counts())
    }

    # Save report
    report_path = output_path / f"{model_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Model report saved to {report_path}")
    return report


# Data validation utilities
def validate_data_quality(df: pd.DataFrame, target_column: str = None):
    """Validate data quality and return quality report."""
    quality_report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }

    # Check for constant columns
    constant_columns = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_columns.append(col)
    quality_report['constant_columns'] = constant_columns

    # Check for high cardinality categorical columns
    high_cardinality_columns = []
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > 50:  # Threshold for high cardinality
            high_cardinality_columns.append({
                'column': col,
                'unique_values': df[col].nunique()
            })
    quality_report['high_cardinality_columns'] = high_cardinality_columns

    # Target variable analysis
    if target_column and target_column in df.columns:
        target_analysis = {
            'unique_values': df[target_column].nunique(),
            'value_counts': df[target_column].value_counts().to_dict(),
            'missing_values': df[target_column].isnull().sum()
        }
        quality_report['target_analysis'] = target_analysis

    return quality_report


def suggest_preprocessing_steps(quality_report: Dict[str, Any]):
    """Suggest preprocessing steps based on data quality report."""
    suggestions = []

    # Missing values
    missing = quality_report.get('missing_values', {})
    if any(count > 0 for count in missing.values()):
        suggestions.append("Handle missing values using imputation or removal")

    # Duplicate rows
    if quality_report.get('duplicate_rows', 0) > 0:
        suggestions.append("Remove duplicate rows")

    # Constant columns
    if quality_report.get('constant_columns'):
        suggestions.append("Remove constant columns as they provide no information")

    # High cardinality columns
    if quality_report.get('high_cardinality_columns'):
        suggestions.append("Consider encoding or reducing cardinality of categorical columns")

    # Target imbalance
    target_analysis = quality_report.get('target_analysis', {})
    if target_analysis:
        value_counts = target_analysis.get('value_counts', {})
        if len(value_counts) > 1:
            min_count = min(value_counts.values())
            max_count = max(value_counts.values())
            if max_count / min_count > 3:  # Imbalance threshold
                suggestions.append("Consider balancing the target variable")

    return suggestions


# Configuration utilities
def create_model_config_template():
    """Create template configuration for models."""
    config_template = {
        "data": {
            "target_column": "gfp_status",
            "test_size": 0.2,
            "validation_size": 0.2,
            "random_state": 42,
            "balance_data": False,
            "scaling": False
        },
        "models": {
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8
            },
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2
            },
            "logistic_regression": {
                "max_iter": 1000,
                "solver": "liblinear"
            }
        },
        "evaluation": {
            "metrics": ["accuracy", "f1", "precision", "recall", "roc_auc"],
            "cross_validation_folds": 5,
            "create_plots": True,
            "save_models": True
        },
        "output": {
            "models_dir": "models",
            "reports_dir": "reports",
            "plots_dir": "plots"
        }
    }

    return config_template


def load_config(config_path: str):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


# Logging utilities
def setup_model_logging(log_level: str = "INFO", log_file: str = "model_training.log"):
    """Setup logging for model training."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_dir / log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


# Performance monitoring utilities
class ModelPerformanceMonitor:
    """Monitor model performance over time."""

    def __init__(self, model_name: str, log_file: str = None):
        self.model_name = model_name
        self.predictions = []
        self.log_file = log_file

        if log_file:
            self.log_path = Path("logs") / log_file
            self.log_path.parent.mkdir(exist_ok=True)

    def log_prediction(self, prediction: Any, confidence: float = None,
                      true_label: Any = None, metadata: Dict[str, Any] = None):
        """Log a single prediction."""
        log_entry = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_name': self.model_name,
            'prediction': prediction,
            'confidence': confidence,
            'true_label': true_label,
            'metadata': metadata or {}
        }

        self.predictions.append(log_entry)

        # Write to file if specified
        if self.log_file:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

    def get_performance_summary(self, window_size: int = 100):
        """Get performance summary for recent predictions."""
        if not self.predictions:
            return {}

        recent_predictions = self.predictions[-window_size:]

        # Calculate metrics if true labels are available
        predictions_with_labels = [p for p in recent_predictions if p['true_label'] is not None]

        if predictions_with_labels:
            y_true = [p['true_label'] for p in predictions_with_labels]
            y_pred = [p['prediction'] for p in predictions_with_labels]

            metrics = calculate_comprehensive_metrics(y_true, y_pred)

            return {
                'total_predictions': len(recent_predictions),
                'predictions_with_labels': len(predictions_with_labels),
                'metrics': metrics,
                'average_confidence': np.mean([p['confidence'] for p in recent_predictions if p['confidence'] is not None])
            }

        return {
            'total_predictions': len(recent_predictions),
            'predictions_with_labels': 0,
            'average_confidence': np.mean([p['confidence'] for p in recent_predictions if p['confidence'] is not None])
        }


# Export utilities
def export_model_for_deployment(model, model_name: str, export_format: str = "onnx"):
    """Export model for deployment in different formats."""
    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)

    if export_format.lower() == "onnx":
        try:
            import torch.onnx
            if hasattr(model, 'state_dict'):  # PyTorch model
                # This would need actual input shape information
                dummy_input = torch.randn(1, 3, 224, 224)  # Example for image model
                export_path = export_dir / f"{model_name}.onnx"
                torch.onnx.export(model, dummy_input, export_path)
                logger.info(f"Model exported to ONNX format: {export_path}")
                return str(export_path)
        except ImportError:
            logger.warning("ONNX export not available")

    elif export_format.lower() == "pickle":
        export_path = export_dir / f"{model_name}.pkl"
        joblib.dump(model, export_path)
        logger.info(f"Model exported to pickle format: {export_path}")
        return str(export_path)

    else:
        raise ValueError(f"Unsupported export format: {export_format}")


# Testing utilities
def create_synthetic_data(n_samples: int = 1000, n_features: int = 10,
                         n_classes: int = 2, random_state: int = 42):
    """Create synthetic data for testing."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        random_state=random_state
    )

    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    return X_df, y_series