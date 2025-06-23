"""
Utilities for XAI module including configuration, saving/loading explanations.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExplanationConfig:
    """Configuration for explanation generation."""
    
    # General settings
    save_explanations: bool = True
    save_plots: bool = True
    output_dir: str = "explanations"
    
    # Image explainer settings
    image_methods: List[str] = None
    target_layer: str = "layer4"  # For Grad-CAM
    noise_level: float = 0.15  # For SmoothGrad
    n_samples: int = 50  # For SmoothGrad
    n_steps: int = 50  # For Integrated Gradients
    
    # Tabular explainer settings
    tabular_methods: List[str] = None
    shap_explainer_type: str = "auto"
    lime_num_features: int = 10
    lime_num_samples: int = 5000
    permutation_n_repeats: int = 10
    pdp_grid_resolution: int = 100
    
    def __post_init__(self):
        """Set default values for list fields."""
        if self.image_methods is None:
            self.image_methods = [
                "VanillaSaliency",
                "GuidedBackpropagation",
                "GradCAM",
                "IntegratedGradients"
            ]
        
        if self.tabular_methods is None:
            self.tabular_methods = [
                "SHAP",
                "PermutationImportance",
                "FeatureImportance"
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExplanationConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: Union[str, Path]):
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ExplanationConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def save_explanation(explanation: Dict[str, Any], 
                    filepath: Union[str, Path],
                    format: str = 'pickle') -> None:
    """
    Save explanation to file.
    
    Args:
        explanation: Explanation dictionary to save
        filepath: Path to save the explanation
        format: Save format ('pickle', 'json', 'npz')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'pickle':
        with open(filepath.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(explanation, f)
    
    elif format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        json_explanation = _convert_numpy_to_list(explanation)
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(json_explanation, f, indent=2)
    
    elif format == 'npz':
        # Save numpy arrays separately
        arrays = {}
        metadata = {}
        
        for key, value in explanation.items():
            if isinstance(value, np.ndarray):
                arrays[key] = value
            else:
                metadata[key] = value
        
        np.savez(filepath.with_suffix('.npz'), **arrays)
        
        # Save metadata separately
        with open(filepath.with_suffix('_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    else:
        raise ValueError("Format must be 'pickle', 'json', or 'npz'")
    
    logger.info(f"Explanation saved to {filepath}")


def load_explanation(filepath: Union[str, Path],
                    format: str = 'pickle') -> Dict[str, Any]:
    """
    Load explanation from file.
    
    Args:
        filepath: Path to the explanation file
        format: Load format ('pickle', 'json', 'npz')
        
    Returns:
        Loaded explanation dictionary
    """
    filepath = Path(filepath)
    
    if format == 'pickle':
        with open(filepath.with_suffix('.pkl'), 'rb') as f:
            return pickle.load(f)
    
    elif format == 'json':
        with open(filepath.with_suffix('.json'), 'r') as f:
            return json.load(f)
    
    elif format == 'npz':
        # Load numpy arrays
        arrays = dict(np.load(filepath.with_suffix('.npz')))
        
        # Load metadata
        metadata_path = filepath.with_suffix('_metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            arrays.update(metadata)
        
        return arrays
    
    else:
        raise ValueError("Format must be 'pickle', 'json', or 'npz'")


def _convert_numpy_to_list(obj: Any) -> Any:
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_list(item) for item in obj]
    else:
        return obj


def plot_explanations(explanations: Dict[str, Dict[str, Any]], 
                     save_dir: Optional[Union[str, Path]] = None,
                     figsize: tuple = (15, 10)) -> plt.Figure:
    """
    Create a comprehensive plot of multiple explanations.
    
    Args:
        explanations: Dictionary of explanations from different methods
        save_dir: Directory to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_methods = len(explanations)
    if n_methods == 0:
        raise ValueError("No explanations provided")
    
    # Determine layout
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_methods == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (method_name, explanation) in enumerate(explanations.items()):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            break
        
        _plot_single_explanation(explanation, method_name, ax)
    
    # Hide unused subplots
    for i in range(n_methods, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"explanations_comparison_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Explanation plot saved to {save_path}")
    
    return fig


def _plot_single_explanation(explanation: Dict[str, Any], 
                           method_name: str, ax: plt.Axes) -> None:
    """Plot a single explanation on the given axes."""
    
    method = explanation.get('method', method_name)
    
    if method == 'SHAP':
        _plot_shap_explanation(explanation, ax)
    elif method == 'Permutation Importance':
        _plot_importance_explanation(explanation, ax, method_name)
    elif method == 'Feature Importance':
        _plot_importance_explanation(explanation, ax, method_name)
    elif method == 'Coefficients':
        _plot_coefficients_explanation(explanation, ax)
    else:
        # Generic plotting for other methods
        _plot_generic_explanation(explanation, ax, method_name)


def _plot_shap_explanation(explanation: Dict[str, Any], ax: plt.Axes) -> None:
    """Plot SHAP explanation."""
    shap_values = explanation.get('shap_values')
    feature_names = explanation.get('feature_names')
    
    if shap_values is not None and len(shap_values.shape) == 2:
        # Take mean absolute SHAP values across samples
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        
        if feature_names is not None:
            indices = np.argsort(mean_shap)[::-1][:10]  # Top 10 features
            ax.barh(range(len(indices)), mean_shap[indices])
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
        else:
            ax.bar(range(len(mean_shap)), mean_shap)
        
        ax.set_title('SHAP Feature Importance')
        ax.set_xlabel('Mean |SHAP Value|')
    else:
        ax.text(0.5, 0.5, 'SHAP values not available\nfor plotting', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('SHAP')


def _plot_importance_explanation(explanation: Dict[str, Any], 
                               ax: plt.Axes, method_name: str) -> None:
    """Plot feature importance explanation."""
    if 'importances_mean' in explanation:
        importances = explanation['importances_mean']
    elif 'feature_importances' in explanation:
        importances = explanation['feature_importances']
    else:
        ax.text(0.5, 0.5, f'{method_name} values\nnot available for plotting', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(method_name)
        return
    
    feature_names = explanation.get('feature_names')
    indices = np.argsort(importances)[::-1][:10]  # Top 10 features
    
    if feature_names is not None:
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
    else:
        ax.bar(range(len(indices)), importances[indices])
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels([f'F{i}' for i in indices])
    
    ax.set_title(method_name)
    ax.set_xlabel('Importance')


def _plot_coefficients_explanation(explanation: Dict[str, Any], ax: plt.Axes) -> None:
    """Plot coefficients explanation."""
    coefficients = explanation.get('coefficients')
    feature_names = explanation.get('feature_names')
    
    if coefficients is not None:
        indices = np.argsort(np.abs(coefficients))[::-1][:10]  # Top 10 features
        colors = ['red' if c < 0 else 'blue' for c in coefficients[indices]]
        
        if feature_names is not None:
            ax.barh(range(len(indices)), coefficients[indices], color=colors)
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
        else:
            ax.bar(range(len(indices)), coefficients[indices], color=colors)
            ax.set_xticks(range(len(indices)))
            ax.set_xticklabels([f'F{i}' for i in indices])
        
        ax.set_title('Model Coefficients')
        ax.set_xlabel('Coefficient Value')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Coefficients not available\nfor plotting', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Coefficients')


def _plot_generic_explanation(explanation: Dict[str, Any], 
                            ax: plt.Axes, method_name: str) -> None:
    """Generic plotting for explanations that don't have specific plotting logic."""
    ax.text(0.5, 0.5, f'{method_name}\nExplanation Available\n(Use specific plotting method)', 
            ha='center', va='center', transform=ax.transAxes)
    ax.set_title(method_name)


def create_explanation_summary(explanations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a summary of explanations for reporting.
    
    Args:
        explanations: Dictionary of explanations from different methods
        
    Returns:
        Summary dictionary
    """
    summary = {
        'timestamp': datetime.now().isoformat(),
        'methods_used': list(explanations.keys()),
        'summary_stats': {}
    }
    
    for method_name, explanation in explanations.items():
        method_summary = {
            'method': explanation.get('method', method_name),
            'available_keys': list(explanation.keys())
        }
        
        # Add method-specific summaries
        if 'shap_values' in explanation:
            shap_values = explanation['shap_values']
            if isinstance(shap_values, np.ndarray):
                method_summary['shap_stats'] = {
                    'shape': shap_values.shape,
                    'mean_abs_shap': float(np.mean(np.abs(shap_values))),
                    'max_abs_shap': float(np.max(np.abs(shap_values)))
                }
        
        if 'feature_importances' in explanation:
            importances = explanation['feature_importances']
            method_summary['importance_stats'] = {
                'top_feature_idx': int(np.argmax(importances)),
                'max_importance': float(np.max(importances)),
                'min_importance': float(np.min(importances))
            }
        
        summary['summary_stats'][method_name] = method_summary
    
    return summary


def save_explanation_report(explanations: Dict[str, Dict[str, Any]],
                          output_dir: Union[str, Path],
                          include_plots: bool = True) -> None:
    """
    Save a comprehensive explanation report.
    
    Args:
        explanations: Dictionary of explanations
        output_dir: Output directory
        include_plots: Whether to include plots in the report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save individual explanations
    for method_name, explanation in explanations.items():
        save_explanation(
            explanation, 
            output_dir / f"{method_name}_{timestamp}",
            format='pickle'
        )
    
    # Save summary
    summary = create_explanation_summary(explanations)
    with open(output_dir / f"explanation_summary_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save plots
    if include_plots:
        try:
            plot_explanations(explanations, save_dir=output_dir)
        except Exception as e:
            logger.warning(f"Could not generate explanation plots: {str(e)}")
    
    logger.info(f"Explanation report saved to {output_dir}")


def load_explanation_report(report_dir: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    Load a complete explanation report.
    
    Args:
        report_dir: Directory containing the report files
        
    Returns:
        Dictionary of loaded explanations
    """
    report_dir = Path(report_dir)
    explanations = {}
    
    # Load all pickle files
    for pkl_file in report_dir.glob("*.pkl"):
        method_name = pkl_file.stem.split('_')[0]  # Extract method name before timestamp
        try:
            explanations[method_name] = load_explanation(pkl_file.with_suffix(''), format='pickle')
        except Exception as e:
            logger.warning(f"Could not load explanation from {pkl_file}: {str(e)}")
    
    return explanations