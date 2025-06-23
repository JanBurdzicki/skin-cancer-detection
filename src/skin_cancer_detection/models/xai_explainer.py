"""
Explainable AI (XAI) module for skin cancer detection models.

This module provides comprehensive explainability methods including SHAP, LIME,
feature importance, and visualization tools for both tabular and image models.
"""

import logging
import warnings
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    from lime.lime_image import LimeImageExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from captum.attr import IntegratedGradients, GradCAM, Saliency
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class XAIConfig:
    """Configuration for XAI explanations."""

    # General settings
    output_dir: str = "explanations"
    save_plots: bool = True

    # SHAP settings
    use_shap: bool = True
    shap_sample_size: int = 100
    shap_background_size: int = 50

    # LIME settings
    use_lime: bool = True
    lime_sample_size: int = 10
    lime_num_features: int = 10

    # Feature importance settings
    use_feature_importance: bool = True
    top_features: int = 15

    # Image explanation settings
    use_gradcam: bool = True
    use_saliency: bool = True
    image_sample_size: int = 5

    # Visualization settings
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    color_palette: str = "viridis"


class TabularExplainer:
    """Explainer for tabular models."""

    def __init__(self, model, X_train: pd.DataFrame, config: XAIConfig = None):
        self.model = model
        self.X_train = X_train
        self.config = config or XAIConfig()
        self.feature_names = X_train.columns.tolist()

        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None

        self._setup_explainers()

    def _setup_explainers(self):
        """Setup SHAP and LIME explainers."""
        try:
            # Setup SHAP explainer
            if self.config.use_shap and SHAP_AVAILABLE:
                if hasattr(self.model, 'predict_proba'):
                    # For tree-based models, use TreeExplainer if possible
                    if hasattr(self.model.model, 'get_booster'):  # XGBoost
                        self.shap_explainer = shap.TreeExplainer(self.model.model)
                    elif hasattr(self.model.model, '_Booster'):  # LightGBM
                        self.shap_explainer = shap.TreeExplainer(self.model.model)
                    else:
                        # Use KernelExplainer for other models
                        background = shap.sample(self.X_train, self.config.shap_background_size)
                        self.shap_explainer = shap.KernelExplainer(
                            self.model.predict_proba, background
                        )
                else:
                    # For models without predict_proba
                    background = shap.sample(self.X_train, self.config.shap_background_size)
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict, background
                    )

                logger.info("SHAP explainer initialized successfully")

            # Setup LIME explainer
            if self.config.use_lime and LIME_AVAILABLE:
                self.lime_explainer = LimeTabularExplainer(
                    self.X_train.values,
                    feature_names=self.feature_names,
                    class_names=['negative', 'positive'],
                    mode='classification'
                )
                logger.info("LIME explainer initialized successfully")

        except Exception as e:
            logger.warning(f"Error setting up explainers: {e}")

    def explain_instance(self, X_instance: pd.DataFrame, instance_idx: int = 0):
        """Explain a single instance."""
        explanations = {}

        # SHAP explanation
        if self.shap_explainer is not None:
            try:
                if hasattr(self.shap_explainer, 'shap_values'):
                    shap_values = self.shap_explainer.shap_values(X_instance.iloc[[instance_idx]])
                else:
                    shap_values = self.shap_explainer(X_instance.iloc[[instance_idx]])

                explanations['shap'] = {
                    'values': shap_values,
                    'feature_names': self.feature_names
                }
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")

        # LIME explanation
        if self.lime_explainer is not None:
            try:
                instance_values = X_instance.iloc[instance_idx].values
                lime_explanation = self.lime_explainer.explain_instance(
                    instance_values,
                    self.model.predict_proba,
                    num_features=self.config.lime_num_features
                )

                explanations['lime'] = lime_explanation
            except Exception as e:
                logger.warning(f"LIME explanation failed: {e}")

        return explanations

    def explain_dataset(self, X_explain: pd.DataFrame):
        """Explain multiple instances."""
        sample_size = min(len(X_explain), self.config.shap_sample_size)
        X_sample = X_explain.sample(n=sample_size, random_state=42)

        explanations = {
            'shap_values': [],
            'lime_explanations': [],
            'feature_importance': None
        }

        # SHAP explanations
        if self.shap_explainer is not None:
            try:
                if hasattr(self.shap_explainer, 'shap_values'):
                    shap_values = self.shap_explainer.shap_values(X_sample)
                else:
                    shap_values = self.shap_explainer(X_sample)

                explanations['shap_values'] = shap_values
                logger.info(f"Generated SHAP explanations for {len(X_sample)} instances")
            except Exception as e:
                logger.warning(f"SHAP batch explanation failed: {e}")

        # LIME explanations (for a subset)
        if self.lime_explainer is not None:
            lime_sample_size = min(self.config.lime_sample_size, len(X_sample))
            X_lime_sample = X_sample.sample(n=lime_sample_size, random_state=42)

            for idx, (_, row) in enumerate(X_lime_sample.iterrows()):
                try:
                    lime_explanation = self.lime_explainer.explain_instance(
                        row.values,
                        self.model.predict_proba,
                        num_features=self.config.lime_num_features
                    )
                    explanations['lime_explanations'].append(lime_explanation)
                except Exception as e:
                    logger.warning(f"LIME explanation failed for instance {idx}: {e}")

        # Feature importance
        if self.config.use_feature_importance:
            try:
                feature_importance = self.model.get_feature_importance()
                explanations['feature_importance'] = feature_importance
            except Exception as e:
                logger.warning(f"Feature importance extraction failed: {e}")

        return explanations

    def create_visualizations(self, explanations: Dict[str, Any], X_explain: pd.DataFrame):
        """Create explanation visualizations."""
        plots = {}

        # Feature importance plot
        if explanations.get('feature_importance'):
            fig = self._plot_feature_importance(explanations['feature_importance'])
            plots['feature_importance'] = fig

            if self.config.save_plots:
                fig.savefig(self.output_dir / "feature_importance.png",
                           dpi=self.config.dpi, bbox_inches='tight')

        # SHAP summary plot
        if explanations.get('shap_values') is not None:
            try:
                fig = self._plot_shap_summary(explanations['shap_values'], X_explain)
                plots['shap_summary'] = fig

                if self.config.save_plots:
                    fig.savefig(self.output_dir / "shap_summary.png",
                               dpi=self.config.dpi, bbox_inches='tight')
            except Exception as e:
                logger.warning(f"SHAP summary plot failed: {e}")

        # LIME explanations plot
        if explanations.get('lime_explanations'):
            try:
                fig = self._plot_lime_explanations(explanations['lime_explanations'])
                plots['lime_explanations'] = fig

                if self.config.save_plots:
                    fig.savefig(self.output_dir / "lime_explanations.png",
                               dpi=self.config.dpi, bbox_inches='tight')
            except Exception as e:
                logger.warning(f"LIME explanations plot failed: {e}")

        return plots

    def _plot_feature_importance(self, feature_importance: Dict[str, float]):
        """Plot feature importance."""
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(),
                               key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_features[:self.config.top_features]

        features, importance = zip(*top_features)

        fig, ax = plt.subplots(figsize=self.config.figure_size)

        # Create horizontal bar plot
        bars = ax.barh(range(len(features)), importance,
                      color=plt.cm.get_cmap(self.config.color_palette)(
                          np.linspace(0, 1, len(features))))

        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {len(features)} Feature Importance')
        ax.invert_yaxis()

        # Add value labels on bars
        for i, (bar, imp) in enumerate(zip(bars, importance)):
            ax.text(bar.get_width() + 0.01 * max(importance),
                   bar.get_y() + bar.get_height()/2,
                   f'{imp:.3f}', va='center', ha='left')

        plt.tight_layout()
        return fig

    def _plot_shap_summary(self, shap_values, X_explain: pd.DataFrame):
        """Plot SHAP summary."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        try:
            # Handle different SHAP value formats
            if isinstance(shap_values, list) and len(shap_values) > 1:
                # Multi-class case - use positive class
                values = shap_values[1]
            elif isinstance(shap_values, np.ndarray):
                values = shap_values
            else:
                values = shap_values

            # Create summary plot
            sample_size = min(len(X_explain), self.config.shap_sample_size)
            X_sample = X_explain.sample(n=sample_size, random_state=42)

            shap.summary_plot(values, X_sample,
                            feature_names=self.feature_names,
                            show=False, ax=ax)
            ax.set_title('SHAP Feature Importance Summary')

        except Exception as e:
            logger.warning(f"SHAP summary plot creation failed: {e}")
            # Fallback to simple feature importance
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            feature_imp = dict(zip(self.feature_names, mean_shap))
            return self._plot_feature_importance(feature_imp)

        plt.tight_layout()
        return fig

    def _plot_lime_explanations(self, lime_explanations: List):
        """Plot LIME explanations."""
        if not lime_explanations:
            return None

        n_explanations = len(lime_explanations)
        fig, axes = plt.subplots(n_explanations, 1,
                                figsize=(self.config.figure_size[0],
                                        self.config.figure_size[1] * n_explanations // 2))

        if n_explanations == 1:
            axes = [axes]

        for i, explanation in enumerate(lime_explanations):
            try:
                # Get explanation data
                exp_data = explanation.as_list()
                features, weights = zip(*exp_data)

                # Create bar plot
                colors = ['red' if w < 0 else 'green' for w in weights]
                axes[i].barh(range(len(features)), weights, color=colors, alpha=0.7)
                axes[i].set_yticks(range(len(features)))
                axes[i].set_yticklabels(features)
                axes[i].set_xlabel('LIME Weight')
                axes[i].set_title(f'LIME Explanation {i+1}')
                axes[i].invert_yaxis()

                # Add vertical line at x=0
                axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.3)

            except Exception as e:
                logger.warning(f"LIME explanation plot {i} failed: {e}")

        plt.tight_layout()
        return fig


class ImageExplainer:
    """Explainer for image models."""

    def __init__(self, model, config: XAIConfig = None):
        self.model = model
        self.config = config or XAIConfig()

        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize explainers
        self.gradcam_explainer = None
        self.saliency_explainer = None
        self.lime_explainer = None

        self._setup_explainers()

    def _setup_explainers(self):
        """Setup image explainers."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - image explanations disabled")
            return

        try:
            # Setup GradCAM
            if self.config.use_gradcam and CAPTUM_AVAILABLE:
                # Find the last convolutional layer
                target_layer = None
                for name, module in self.model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        target_layer = module

                if target_layer is not None:
                    self.gradcam_explainer = GradCAM(self.model, target_layer)
                    logger.info("GradCAM explainer initialized")

            # Setup Saliency
            if self.config.use_saliency and CAPTUM_AVAILABLE:
                self.saliency_explainer = Saliency(self.model)
                logger.info("Saliency explainer initialized")

            # Setup LIME for images
            if LIME_AVAILABLE:
                self.lime_explainer = LimeImageExplainer()
                logger.info("LIME image explainer initialized")

        except Exception as e:
            logger.warning(f"Error setting up image explainers: {e}")

    def explain_images(self, images: torch.Tensor, labels: torch.Tensor = None):
        """Explain image predictions."""
        explanations = {}

        # Limit sample size
        sample_size = min(len(images), self.config.image_sample_size)
        sample_images = images[:sample_size]

        # GradCAM explanations
        if self.gradcam_explainer is not None:
            try:
                gradcam_attrs = []
                for img in sample_images:
                    attr = self.gradcam_explainer.attribute(img.unsqueeze(0), target=0)
                    gradcam_attrs.append(attr.squeeze().cpu().numpy())

                explanations['gradcam'] = gradcam_attrs
                logger.info(f"Generated GradCAM explanations for {len(gradcam_attrs)} images")
            except Exception as e:
                logger.warning(f"GradCAM explanation failed: {e}")

        # Saliency explanations
        if self.saliency_explainer is not None:
            try:
                saliency_attrs = []
                for img in sample_images:
                    img.requires_grad_()
                    attr = self.saliency_explainer.attribute(img.unsqueeze(0), target=0)
                    saliency_attrs.append(attr.squeeze().cpu().numpy())

                explanations['saliency'] = saliency_attrs
                logger.info(f"Generated Saliency explanations for {len(saliency_attrs)} images")
            except Exception as e:
                logger.warning(f"Saliency explanation failed: {e}")

        return explanations

    def create_visualizations(self, explanations: Dict[str, Any], original_images: torch.Tensor):
        """Create image explanation visualizations."""
        plots = {}

        n_images = min(len(original_images), self.config.image_sample_size)

        # Create combined visualization
        fig, axes = plt.subplots(3, n_images, figsize=(4*n_images, 12))
        if n_images == 1:
            axes = axes.reshape(-1, 1)

        for i in range(n_images):
            # Original image
            orig_img = original_images[i].cpu().numpy().transpose(1, 2, 0)
            if orig_img.shape[2] == 1:  # Grayscale
                orig_img = orig_img.squeeze(2)
                axes[0, i].imshow(orig_img, cmap='gray')
            else:
                # Normalize for display
                orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
                axes[0, i].imshow(orig_img)

            axes[0, i].set_title(f'Original Image {i+1}')
            axes[0, i].axis('off')

            # GradCAM
            if 'gradcam' in explanations and i < len(explanations['gradcam']):
                gradcam_attr = explanations['gradcam'][i]
                im = axes[1, i].imshow(gradcam_attr, cmap='hot', alpha=0.7)
                axes[1, i].set_title(f'GradCAM {i+1}')
                axes[1, i].axis('off')
            else:
                axes[1, i].text(0.5, 0.5, 'GradCAM\nNot Available',
                               ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].axis('off')

            # Saliency
            if 'saliency' in explanations and i < len(explanations['saliency']):
                saliency_attr = explanations['saliency'][i]
                # Take absolute value and average across channels if needed
                if len(saliency_attr.shape) == 3:
                    saliency_attr = np.mean(np.abs(saliency_attr), axis=0)
                else:
                    saliency_attr = np.abs(saliency_attr)

                axes[2, i].imshow(saliency_attr, cmap='hot')
                axes[2, i].set_title(f'Saliency {i+1}')
                axes[2, i].axis('off')
            else:
                axes[2, i].text(0.5, 0.5, 'Saliency\nNot Available',
                               ha='center', va='center', transform=axes[2, i].transAxes)
                axes[2, i].axis('off')

        plt.tight_layout()
        plots['image_explanations'] = fig

        if self.config.save_plots:
            fig.savefig(self.output_dir / "image_explanations.png",
                       dpi=self.config.dpi, bbox_inches='tight')

        return plots


# Utility functions
def create_tabular_explainer(model, X_train: pd.DataFrame, config: XAIConfig = None):
    """Factory function to create tabular explainer."""
    return TabularExplainer(model, X_train, config)


def create_image_explainer(model, config: XAIConfig = None):
    """Factory function to create image explainer."""
    return ImageExplainer(model, config)


def explain_model(model, model_type: str, data: Any, **kwargs):
    """Quick explanation function."""
    config = XAIConfig(**kwargs)

    if model_type == "tabular":
        X_train, X_explain = data
        explainer = create_tabular_explainer(model, X_train, config)
        explanations = explainer.explain_dataset(X_explain)
        plots = explainer.create_visualizations(explanations, X_explain)
        return explanations, plots

    elif model_type == "image":
        images = data
        explainer = create_image_explainer(model, config)
        explanations = explainer.explain_images(images)
        plots = explainer.create_visualizations(explanations, images)
        return explanations, plots

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_available_explanation_methods():
    """Get list of available explanation methods."""
    methods = {
        'tabular': ['feature_importance'],
        'image': []
    }

    if SHAP_AVAILABLE:
        methods['tabular'].append('shap')

    if LIME_AVAILABLE:
        methods['tabular'].append('lime')
        methods['image'].append('lime')

    if CAPTUM_AVAILABLE:
        methods['image'].extend(['gradcam', 'saliency'])

    return methods