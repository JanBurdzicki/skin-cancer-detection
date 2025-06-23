"""
Tabular explainability methods for skin cancer detection models.

This module implements various model-agnostic and model-specific explainability
methods for tabular data including SHAP, LIME, permutation importance, and more.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import logging

# Import explainability libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available. Install with: pip install lime")

try:
    from treeinterpreter import treeinterpreter as ti
    TREE_INTERPRETER_AVAILABLE = True
except ImportError:
    TREE_INTERPRETER_AVAILABLE = False
    warnings.warn("TreeInterpreter not available. Install with: pip install treeinterpreter")

logger = logging.getLogger(__name__)


class TabularExplainerBase(ABC):
    """Base class for all tabular explainability methods."""

    def __init__(self, model: BaseEstimator, feature_names: Optional[List[str]] = None):
        """
        Initialize the explainer.

        Args:
            model: The trained model to explain
            feature_names: Names of features in the dataset
        """
        self.model = model
        self.feature_names = feature_names

    @abstractmethod
    def explain(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Generate explanation for the input data."""
        pass

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate and preprocess input data."""
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X = X.values
        elif isinstance(X, list):
            X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        return X


class SHAPExplainer(TabularExplainerBase):
    """SHAP (SHapley Additive exPlanations) explainer for all model types."""

    def __init__(self, model: BaseEstimator, X_train: np.ndarray,
                 feature_names: Optional[List[str]] = None,
                 explainer_type: str = "auto"):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model
            X_train: Training data for background distribution
            feature_names: Feature names
            explainer_type: Type of SHAP explainer ('tree', 'linear', 'kernel', 'auto')
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for SHAPExplainer")

        super().__init__(model, feature_names)
        self.X_train = self._validate_input(X_train)
        self.explainer_type = explainer_type
        self.explainer = self._create_explainer()

    def _create_explainer(self):
        """Create appropriate SHAP explainer based on model type."""
        model_name = type(self.model).__name__.lower()

        if self.explainer_type == "auto":
            if any(name in model_name for name in ['forest', 'tree', 'xgb', 'lgb', 'gbm']):
                return shap.TreeExplainer(self.model)
            elif any(name in model_name for name in ['linear', 'logistic']):
                return shap.LinearExplainer(self.model, self.X_train)
            else:
                return shap.KernelExplainer(self.model.predict_proba,
                                         shap.sample(self.X_train, 100))
        elif self.explainer_type == "tree":
            return shap.TreeExplainer(self.model)
        elif self.explainer_type == "linear":
            return shap.LinearExplainer(self.model, self.X_train)
        elif self.explainer_type == "kernel":
            return shap.KernelExplainer(self.model.predict_proba,
                                      shap.sample(self.X_train, 100))
        else:
            raise ValueError("explainer_type must be 'tree', 'linear', 'kernel', or 'auto'")

    def explain(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Generate SHAP explanations."""
        X = self._validate_input(X)

        shap_values = self.explainer.shap_values(X)

        # Handle multi-class case
        if isinstance(shap_values, list):
            # For binary classification, use positive class
            if len(shap_values) == 2:
                shap_values = shap_values[1]
            else:
                # For multi-class, return all classes
                pass

        return {
            'shap_values': shap_values,
            'expected_value': self.explainer.expected_value,
            'feature_names': self.feature_names,
            'method': 'SHAP'
        }

    def plot_summary(self, X: np.ndarray, save_path: Optional[str] = None, **kwargs):
        """Create SHAP summary plot."""
        X = self._validate_input(X)
        explanation = self.explain(X)

        shap.summary_plot(explanation['shap_values'], X,
                         feature_names=self.feature_names, show=False)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return plt.gcf()


class PermutationImportanceExplainer(TabularExplainerBase):
    """Permutation importance explainer (model-agnostic)."""

    def __init__(self, model: BaseEstimator, feature_names: Optional[List[str]] = None):
        super().__init__(model, feature_names)

    def explain(self, X: np.ndarray, y: np.ndarray,
                scoring: str = 'accuracy', n_repeats: int = 10,
                random_state: int = 42, **kwargs) -> Dict[str, Any]:
        """
        Generate permutation importance explanations.

        Args:
            X: Input features
            y: Target values
            scoring: Scoring method
            n_repeats: Number of permutation repeats
            random_state: Random state for reproducibility
        """
        X = self._validate_input(X)

        perm_importance = permutation_importance(
            self.model, X, y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state
        )

        return {
            'importances_mean': perm_importance.importances_mean,
            'importances_std': perm_importance.importances_std,
            'importances': perm_importance.importances,
            'feature_names': self.feature_names,
            'method': 'Permutation Importance'
        }

    def plot_importance(self, X: np.ndarray, y: np.ndarray,
                       save_path: Optional[str] = None, **kwargs):
        """Plot permutation importance."""
        explanation = self.explain(X, y, **kwargs)

        indices = np.argsort(explanation['importances_mean'])[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Permutation Feature Importance")

        feature_names = self.feature_names or [f"Feature {i}" for i in range(len(indices))]

        plt.bar(range(len(indices)), explanation['importances_mean'][indices],
                yerr=explanation['importances_std'][indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return plt.gcf()


class LIMEExplainer(TabularExplainerBase):
    """LIME (Local Interpretable Model-agnostic Explanations) explainer."""

    def __init__(self, model: BaseEstimator, X_train: np.ndarray,
                 feature_names: Optional[List[str]] = None,
                 mode: str = 'classification'):
        """
        Initialize LIME explainer.

        Args:
            model: Trained model
            X_train: Training data
            feature_names: Feature names
            mode: 'classification' or 'regression'
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required for LIMEExplainer")

        super().__init__(model, feature_names)
        self.X_train = self._validate_input(X_train)
        self.mode = mode

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train,
            feature_names=self.feature_names,
            mode=self.mode,
            discretize_continuous=True
        )

    def explain(self, X: np.ndarray, num_features: int = 10,
                num_samples: int = 5000, **kwargs) -> Dict[str, Any]:
        """
        Generate LIME explanations.

        Args:
            X: Input instances to explain
            num_features: Number of features to include in explanation
            num_samples: Number of samples for LIME
        """
        X = self._validate_input(X)

        explanations = []
        for i in range(X.shape[0]):
            if self.mode == 'classification':
                explanation = self.explainer.explain_instance(
                    X[i], self.model.predict_proba,
                    num_features=num_features, num_samples=num_samples
                )
            else:
                explanation = self.explainer.explain_instance(
                    X[i], self.model.predict,
                    num_features=num_features, num_samples=num_samples
                )
            explanations.append(explanation)

        return {
            'explanations': explanations,
            'feature_names': self.feature_names,
            'method': 'LIME'
        }


class PDPExplainer(TabularExplainerBase):
    """Partial Dependence Plots explainer."""

    def __init__(self, model: BaseEstimator, feature_names: Optional[List[str]] = None):
        super().__init__(model, feature_names)

    def explain(self, X: np.ndarray, features: Union[int, List[int], str, List[str]],
                grid_resolution: int = 100, **kwargs) -> Dict[str, Any]:
        """
        Generate partial dependence explanations.

        Args:
            X: Input data
            features: Features to compute PDP for
            grid_resolution: Resolution of the grid
        """
        X = self._validate_input(X)

        if isinstance(features, str):
            if self.feature_names:
                features = [self.feature_names.index(features)]
            else:
                raise ValueError("Feature names required for string feature specification")
        elif isinstance(features, list) and isinstance(features[0], str):
            if self.feature_names:
                features = [self.feature_names.index(f) for f in features]
            else:
                raise ValueError("Feature names required for string feature specification")

        pdp_result = partial_dependence(
            self.model, X, features,
            grid_resolution=grid_resolution,
            kind='average'
        )

        return {
            'partial_dependence': pdp_result.average,
            'grid_values': pdp_result.grid_values,
            'features': features,
            'feature_names': self.feature_names,
            'method': 'Partial Dependence Plot'
        }

    def plot_pdp(self, X: np.ndarray, features: Union[int, List[int], str, List[str]],
                 save_path: Optional[str] = None, **kwargs):
        """Plot partial dependence."""
        explanation = self.explain(X, features, **kwargs)

        if len(explanation['features']) == 1:
            plt.figure(figsize=(8, 6))
            feature_idx = explanation['features'][0]
            feature_name = (self.feature_names[feature_idx] if self.feature_names
                          else f"Feature {feature_idx}")

            plt.plot(explanation['grid_values'][0], explanation['partial_dependence'][0])
            plt.xlabel(feature_name)
            plt.ylabel('Partial Dependence')
            plt.title(f'Partial Dependence Plot: {feature_name}')
            plt.grid(True)
        else:
            # 2D PDP
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(explanation['partial_dependence'],
                          extent=[explanation['grid_values'][0].min(),
                                 explanation['grid_values'][0].max(),
                                 explanation['grid_values'][1].min(),
                                 explanation['grid_values'][1].max()],
                          aspect='auto', origin='lower')
            plt.colorbar(im)

            feature_names = []
            for idx in explanation['features']:
                feature_names.append(self.feature_names[idx] if self.feature_names
                                   else f"Feature {idx}")

            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
            ax.set_title(f'Partial Dependence Plot: {feature_names[0]} vs {feature_names[1]}')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return plt.gcf()


class ICEExplainer(TabularExplainerBase):
    """Individual Conditional Expectation (ICE) explainer."""

    def __init__(self, model: BaseEstimator, feature_names: Optional[List[str]] = None):
        super().__init__(model, feature_names)

    def explain(self, X: np.ndarray, feature: Union[int, str],
                grid_resolution: int = 100, **kwargs) -> Dict[str, Any]:
        """
        Generate ICE explanations.

        Args:
            X: Input data
            feature: Feature to compute ICE for
            grid_resolution: Resolution of the grid
        """
        X = self._validate_input(X)

        if isinstance(feature, str):
            if self.feature_names:
                feature_idx = self.feature_names.index(feature)
            else:
                raise ValueError("Feature names required for string feature specification")
        else:
            feature_idx = feature

        ice_result = partial_dependence(
            self.model, X, [feature_idx],
            grid_resolution=grid_resolution,
            kind='individual'
        )

        return {
            'individual_dependence': ice_result.individual,
            'average_dependence': ice_result.average,
            'grid_values': ice_result.grid_values[0],
            'feature': feature_idx,
            'feature_names': self.feature_names,
            'method': 'Individual Conditional Expectation'
        }

    def plot_ice(self, X: np.ndarray, feature: Union[int, str],
                 save_path: Optional[str] = None, **kwargs):
        """Plot ICE curves."""
        explanation = self.explain(X, feature, **kwargs)

        plt.figure(figsize=(10, 6))

        # Plot individual curves
        for i in range(explanation['individual_dependence'].shape[0]):
            plt.plot(explanation['grid_values'],
                    explanation['individual_dependence'][i],
                    alpha=0.3, color='blue')

        # Plot average curve
        plt.plot(explanation['grid_values'],
                explanation['average_dependence'][0],
                color='red', linewidth=3, label='Average')

        feature_name = (self.feature_names[explanation['feature']] if self.feature_names
                       else f"Feature {explanation['feature']}")

        plt.xlabel(feature_name)
        plt.ylabel('Prediction')
        plt.title(f'ICE Plot: {feature_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return plt.gcf()


class FeatureImportanceExplainer(TabularExplainerBase):
    """Feature importance explainer for tree-based models."""

    def __init__(self, model: BaseEstimator, feature_names: Optional[List[str]] = None):
        super().__init__(model, feature_names)

        # Check if model has feature importance
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")

    def explain(self, **kwargs) -> Dict[str, Any]:
        """Generate feature importance explanations."""
        return {
            'feature_importances': self.model.feature_importances_,
            'feature_names': self.feature_names,
            'method': 'Feature Importance'
        }

    def plot_importance(self, save_path: Optional[str] = None, **kwargs):
        """Plot feature importance."""
        explanation = self.explain(**kwargs)

        indices = np.argsort(explanation['feature_importances'])[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")

        feature_names = self.feature_names or [f"Feature {i}" for i in range(len(indices))]

        plt.bar(range(len(indices)), explanation['feature_importances'][indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return plt.gcf()


class CoefficientsExplainer(TabularExplainerBase):
    """Coefficients explainer for linear models."""

    def __init__(self, model: BaseEstimator, feature_names: Optional[List[str]] = None):
        super().__init__(model, feature_names)

        # Check if model has coefficients
        if not hasattr(model, 'coef_'):
            raise ValueError("Model does not have coef_ attribute")

    def explain(self, **kwargs) -> Dict[str, Any]:
        """Generate coefficient explanations."""
        coefficients = self.model.coef_

        # Handle multi-class case
        if coefficients.ndim > 1:
            coefficients = coefficients[0]  # Use first class for binary

        return {
            'coefficients': coefficients,
            'intercept': getattr(self.model, 'intercept_', 0),
            'feature_names': self.feature_names,
            'method': 'Coefficients'
        }

    def plot_coefficients(self, save_path: Optional[str] = None, **kwargs):
        """Plot model coefficients."""
        explanation = self.explain(**kwargs)

        coefficients = explanation['coefficients']
        indices = np.argsort(np.abs(coefficients))[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Model Coefficients")

        feature_names = self.feature_names or [f"Feature {i}" for i in range(len(indices))]

        colors = ['red' if c < 0 else 'blue' for c in coefficients[indices]]
        plt.bar(range(len(indices)), coefficients[indices], color=colors)
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        plt.ylabel('Coefficient Value')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return plt.gcf()


class TreeInterpreterExplainer(TabularExplainerBase):
    """TreeInterpreter for local feature importance in tree-based models."""

    def __init__(self, model: BaseEstimator, feature_names: Optional[List[str]] = None):
        if not TREE_INTERPRETER_AVAILABLE:
            raise ImportError("TreeInterpreter is required for TreeInterpreterExplainer")

        super().__init__(model, feature_names)

        # Check if model is supported
        model_name = type(model).__name__.lower()
        if not any(name in model_name for name in ['forest', 'tree', 'extratree']):
            raise ValueError("TreeInterpreter only supports tree-based models")

    def explain(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Generate TreeInterpreter explanations."""
        X = self._validate_input(X)

        prediction, bias, contributions = ti.predict(self.model, X)

        return {
            'predictions': prediction,
            'bias': bias,
            'contributions': contributions,
            'feature_names': self.feature_names,
            'method': 'TreeInterpreter'
        }

    def plot_local_importance(self, X: np.ndarray, instance_idx: int = 0,
                             save_path: Optional[str] = None, **kwargs):
        """Plot local feature importance for a specific instance."""
        explanation = self.explain(X, **kwargs)

        contributions = explanation['contributions'][instance_idx]
        indices = np.argsort(np.abs(contributions))[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(f"Local Feature Importance (Instance {instance_idx})")

        feature_names = self.feature_names or [f"Feature {i}" for i in range(len(indices))]

        colors = ['red' if c < 0 else 'blue' for c in contributions[indices]]
        plt.bar(range(len(indices)), contributions[indices], color=colors)
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        plt.ylabel('Contribution')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return plt.gcf()


def create_explanation_report(explainers: List[TabularExplainerBase],
                            X: np.ndarray, y: Optional[np.ndarray] = None,
                            save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a comprehensive explanation report using multiple explainers.

    Args:
        explainers: List of explainer instances
        X: Input data
        y: Target data (required for some explainers)
        save_dir: Directory to save plots

    Returns:
        Dictionary containing all explanations
    """
    report = {}

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for explainer in explainers:
        explainer_name = type(explainer).__name__
        logger.info(f"Generating explanations with {explainer_name}")

        try:
            if isinstance(explainer, PermutationImportanceExplainer) and y is not None:
                explanation = explainer.explain(X, y)
                if save_dir:
                    explainer.plot_importance(X, y,
                                            save_path=save_dir / f"{explainer_name}_plot.png")
            else:
                explanation = explainer.explain(X)

                # Generate plots if available
                if hasattr(explainer, 'plot_summary') and save_dir:
                    explainer.plot_summary(X, save_path=save_dir / f"{explainer_name}_summary.png")
                elif hasattr(explainer, 'plot_importance') and save_dir:
                    explainer.plot_importance(save_path=save_dir / f"{explainer_name}_plot.png")
                elif hasattr(explainer, 'plot_coefficients') and save_dir:
                    explainer.plot_coefficients(save_path=save_dir / f"{explainer_name}_plot.png")

            report[explainer_name] = explanation

        except Exception as e:
            logger.error(f"Error with {explainer_name}: {str(e)}")
            report[explainer_name] = {'error': str(e)}

    return report