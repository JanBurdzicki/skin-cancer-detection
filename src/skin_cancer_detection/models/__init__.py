"""
Models module for skin cancer detection.

This module provides comprehensive training, fine-tuning, validation, and testing
capabilities for both image and tabular data models with wandb integration.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Track which model components are available
_AVAILABLE_COMPONENTS = {
    'image_models': False,
    'tabular_models': False,
    'model_trainer': False,
    'xai_explainer': False,
    'deployment': False,
    'ensemble': False,
    'evaluation': False,
    'hyperparameter_tuning': False,
    'cross_validation': False,
    'utils': False
}

# Image Models
try:
    from .image_models import (
        CNNClassifier,
        ResNet18Classifier,
        ImageModelTrainer,
        ImageModelConfig
    )
    _AVAILABLE_COMPONENTS['image_models'] = True
    logger.debug("Image models loaded successfully")
except ImportError as e:
    logger.debug(f"Image models not available: {e}")
    # Create placeholder classes to avoid import errors
    CNNClassifier = None
    ResNet18Classifier = None
    ImageModelTrainer = None
    ImageModelConfig = None

# Tabular Models
try:
    from .tabular_models import (
        XGBoostClassifier,
        LightGBMClassifier,
        RandomForestClassifier,
        LogisticRegressionClassifier,
        TabularModelTrainer,
        TabularModelConfig
    )
    _AVAILABLE_COMPONENTS['tabular_models'] = True
    logger.debug("Tabular models loaded successfully")
except ImportError as e:
    logger.debug(f"Tabular models not available: {e}")
    # Create placeholder classes
    XGBoostClassifier = None
    LightGBMClassifier = None
    RandomForestClassifier = None
    LogisticRegressionClassifier = None
    TabularModelTrainer = None
    TabularModelConfig = None

# Model Trainer
try:
    from .model_trainer import (
        UnifiedModelTrainer,
        UnifiedTrainingConfig,
        EnsembleModel,
        create_unified_trainer,
        train_model
    )
    _AVAILABLE_COMPONENTS['model_trainer'] = True
    logger.debug("Model trainer loaded successfully")
except ImportError as e:
    logger.debug(f"Model trainer not available: {e}")
    UnifiedModelTrainer = None
    UnifiedTrainingConfig = None
    EnsembleModel = None
    create_unified_trainer = None
    train_model = None

# XAI Explainer
try:
    from .xai_explainer import (
        TabularExplainer,
        ImageExplainer,
        XAIConfig,
        create_tabular_explainer,
        create_image_explainer,
        explain_model
    )
    _AVAILABLE_COMPONENTS['xai_explainer'] = True
    logger.debug("XAI explainer loaded successfully")
except ImportError as e:
    logger.debug(f"XAI explainer not available: {e}")
    TabularExplainer = None
    ImageExplainer = None
    XAIConfig = None
    create_tabular_explainer = None
    create_image_explainer = None
    explain_model = None

# Deployment
try:
    from .deployment import (
        ModelDeployer,
        TabularModelDeployer,
        ImageModelDeployer,
        EnsembleModelDeployer,
        DeploymentConfig,
        create_deployer,
        deploy_model
    )
    _AVAILABLE_COMPONENTS['deployment'] = True
    logger.debug("Deployment tools loaded successfully")
except ImportError as e:
    logger.debug(f"Deployment tools not available: {e}")
    ModelDeployer = None
    TabularModelDeployer = None
    ImageModelDeployer = None
    EnsembleModelDeployer = None
    DeploymentConfig = None
    create_deployer = None
    deploy_model = None

# Ensemble
try:
    from .ensemble import (
        BaseEnsemble,
        VotingEnsemble,
        WeightedAverageEnsemble,
        StackingEnsemble,
        AdaptiveEnsemble,
        EnsembleConfig,
        create_ensemble,
        evaluate_ensemble
    )
    _AVAILABLE_COMPONENTS['ensemble'] = True
    logger.debug("Ensemble methods loaded successfully")
except ImportError as e:
    logger.debug(f"Ensemble methods not available: {e}")
    BaseEnsemble = None
    VotingEnsemble = None
    WeightedAverageEnsemble = None
    StackingEnsemble = None
    AdaptiveEnsemble = None
    EnsembleConfig = None
    create_ensemble = None
    evaluate_ensemble = None

# Evaluation Tools
try:
    from .evaluation import (
        ModelEvaluator,
        EvaluationConfig,
        evaluate_model,
        compare_models,
        calculate_metrics
    )
    _AVAILABLE_COMPONENTS['evaluation'] = True
    logger.debug("Evaluation tools loaded successfully")
except ImportError as e:
    logger.debug(f"Evaluation tools not available: {e}")
    ModelEvaluator = None
    EvaluationConfig = None
    evaluate_model = None
    compare_models = None
    calculate_metrics = None

# Hyperparameter Tuning
try:
    from .hyperparameter_tuning import (
        HyperparameterTuningConfig,
        GridSearchTuner,
        RandomSearchTuner,
        BayesianTuner,
        OptunaTuner,
        create_tuner,
        tune_model
    )
    _AVAILABLE_COMPONENTS['hyperparameter_tuning'] = True
    logger.debug("Hyperparameter tuning tools loaded successfully")
except ImportError as e:
    logger.debug(f"Hyperparameter tuning tools not available: {e}")
    HyperparameterTuningConfig = None
    GridSearchTuner = None
    RandomSearchTuner = None
    BayesianTuner = None
    OptunaTuner = None
    create_tuner = None
    tune_model = None

# Cross Validation
try:
    from .cross_validation import (
        CrossValidator,
        CrossValidationConfig,
        TimeSeriesCrossValidator,
        cross_validate_model,
        compare_models_cv,
        nested_cv
    )
    _AVAILABLE_COMPONENTS['cross_validation'] = True
    logger.debug("Cross validation tools loaded successfully")
except ImportError as e:
    logger.debug(f"Cross validation tools not available: {e}")
    CrossValidator = None
    CrossValidationConfig = None
    TimeSeriesCrossValidator = None
    cross_validate_model = None
    compare_models_cv = None
    nested_cv = None

# Utilities
try:
    from .utils import (
        preprocess_tabular_data,
        calculate_comprehensive_metrics,
        create_confusion_matrix_plot,
        create_roc_curve_plot,
        save_model_artifacts,
        load_model_artifacts,
        create_model_report,
        validate_data_quality,
        ModelPerformanceMonitor
    )
    _AVAILABLE_COMPONENTS['utils'] = True
    logger.debug("Model utilities loaded successfully")
except ImportError as e:
    logger.debug(f"Model utilities not available: {e}")
    preprocess_tabular_data = None
    calculate_comprehensive_metrics = None
    create_confusion_matrix_plot = None
    create_roc_curve_plot = None
    save_model_artifacts = None
    load_model_artifacts = None
    create_model_report = None
    validate_data_quality = None
    ModelPerformanceMonitor = None


def get_available_components() -> Dict[str, bool]:
    """
    Get information about which model components are available.

    Returns:
        Dictionary mapping component names to availability status
    """
    return _AVAILABLE_COMPONENTS.copy()


def is_component_available(component_name: str) -> bool:
    """
    Check if a specific model component is available.

    Args:
        component_name: Name of the component to check

    Returns:
        True if component is available, False otherwise
    """
    return _AVAILABLE_COMPONENTS.get(component_name, False)


def get_available_models() -> Dict[str, Any]:
    """
    Get a dictionary of available model classes.

    Returns:
        Dictionary of available model classes
    """
    available_models = {}

    if _AVAILABLE_COMPONENTS['image_models']:
        available_models.update({
            'CNNClassifier': CNNClassifier,
            'ResNet18Classifier': ResNet18Classifier,
            'ImageModelTrainer': ImageModelTrainer,
            'ImageModelConfig': ImageModelConfig
        })

    if _AVAILABLE_COMPONENTS['tabular_models']:
        available_models.update({
            'XGBoostClassifier': XGBoostClassifier,
            'LightGBMClassifier': LightGBMClassifier,
            'RandomForestClassifier': RandomForestClassifier,
            'LogisticRegressionClassifier': LogisticRegressionClassifier,
            'TabularModelTrainer': TabularModelTrainer,
            'TabularModelConfig': TabularModelConfig
        })

    return available_models


def print_component_status():
    """Print the availability status of all model components."""
    print("Model Components Availability:")
    print("=" * 40)
    for component, available in _AVAILABLE_COMPONENTS.items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"{component:20} : {status}")
    print("=" * 40)


# Build __all__ dynamically based on what's available
__all__ = [
    # Utility functions
    'get_available_components',
    'is_component_available',
    'get_available_models',
    'print_component_status'
]

# Add available model classes to __all__
if _AVAILABLE_COMPONENTS['image_models']:
    __all__.extend([
        'CNNClassifier',
        'ResNet18Classifier',
        'ImageModelTrainer',
        'ImageModelConfig'
    ])

if _AVAILABLE_COMPONENTS['tabular_models']:
    __all__.extend([
        'XGBoostClassifier',
        'LightGBMClassifier',
        'RandomForestClassifier',
        'LogisticRegressionClassifier',
        'TabularModelTrainer',
        'TabularModelConfig'
    ])

if _AVAILABLE_COMPONENTS['evaluation']:
    __all__.extend([
        'ModelEvaluator',
        'EvaluationMetrics',
        'compute_all_metrics',
        'plot_confusion_matrix',
        'plot_roc_curves',
        'plot_precision_recall_curves'
    ])

if _AVAILABLE_COMPONENTS['hyperparameter_tuning']:
    __all__.extend([
        'OptunaHyperparameterTuner',
        'HyperparameterConfig',
        'create_optuna_study'
    ])

if _AVAILABLE_COMPONENTS['cross_validation']:
    __all__.extend([
        'CrossValidator',
        'CrossValidationConfig',
        'stratified_cross_validation'
    ])

if _AVAILABLE_COMPONENTS['utils']:
    __all__.extend([
        'ModelConfig',
        'save_model_artifact',
        'load_model_artifact',
        'setup_wandb',
        'log_metrics_to_wandb',
        'handle_class_imbalance'
    ])

# Log summary of loaded components
available_count = sum(_AVAILABLE_COMPONENTS.values())
total_count = len(_AVAILABLE_COMPONENTS)
logger.info(f"Models module initialized: {available_count}/{total_count} components available")