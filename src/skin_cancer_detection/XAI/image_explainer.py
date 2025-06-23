"""
Image explainability methods for skin cancer detection models.

This module implements various gradient-based and perturbation-based explainability
methods specifically for image classification models.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from captum.attr import (
    Saliency,
    GuidedBackprop,
    DeepLift,
    IntegratedGradients,
    NoiseTunnel
)
from captum.attr import LayerGradCam, GuidedGradCam
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import logging

logger = logging.getLogger(__name__)


class ImageExplainerBase(ABC):
    """Base class for all image explainability methods."""

    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Initialize the explainer.

        Args:
            model: The PyTorch model to explain
            device: Device to run computations on
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    @abstractmethod
    def explain(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generate explanation for the input."""
        pass

    def preprocess_input(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess input tensor for explanation."""
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        return input_tensor.to(self.device).requires_grad_(True)

    def postprocess_attribution(self, attribution: torch.Tensor) -> np.ndarray:
        """Postprocess attribution map for visualization."""
        if attribution.dim() == 4:
            attribution = attribution.squeeze(0)

        # Convert to numpy
        if isinstance(attribution, torch.Tensor):
            attribution = attribution.detach().cpu().numpy()

        # Handle multi-channel attributions
        if attribution.shape[0] > 1:
            attribution = np.mean(attribution, axis=0)
        else:
            attribution = attribution[0]

        return attribution


class VanillaSaliency(ImageExplainerBase):
    """Vanilla Saliency method using simple gradients."""

    def __init__(self, model: nn.Module, device: torch.device = None):
        super().__init__(model, device)
        self.saliency = Saliency(self.model)

    def explain(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate vanilla saliency map.

        Args:
            input_tensor: Input image tensor
            target_class: Target class for explanation (None for predicted class)

        Returns:
            Saliency map as numpy array
        """
        input_tensor = self.preprocess_input(input_tensor)

        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()

        attribution = self.saliency.attribute(input_tensor, target=target_class)
        return self.postprocess_attribution(attribution)


class GuidedBackpropagation(ImageExplainerBase):
    """Guided Backpropagation method."""

    def __init__(self, model: nn.Module, device: torch.device = None):
        super().__init__(model, device)
        self.guided_bp = GuidedBackprop(self.model)

    def explain(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generate guided backpropagation explanation."""
        input_tensor = self.preprocess_input(input_tensor)

        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()

        attribution = self.guided_bp.attribute(input_tensor, target=target_class)
        return self.postprocess_attribution(attribution)


class DeepLIFTExplainer(ImageExplainerBase):
    """DeepLIFT explainability method."""

    def __init__(self, model: nn.Module, device: torch.device = None):
        super().__init__(model, device)
        self.deeplift = DeepLift(self.model)

    def explain(self, input_tensor: torch.Tensor, target_class: Optional[int] = None,
                baseline: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Generate DeepLIFT explanation.

        Args:
            input_tensor: Input image tensor
            target_class: Target class for explanation
            baseline: Baseline for comparison (default: zeros)

        Returns:
            Attribution map as numpy array
        """
        input_tensor = self.preprocess_input(input_tensor)

        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()

        attribution = self.deeplift.attribute(input_tensor, baselines=baseline, target=target_class)
        return self.postprocess_attribution(attribution)


class GradCAMExplainer(ImageExplainerBase):
    """Grad-CAM and Guided Grad-CAM explainability method."""

    def __init__(self, model: nn.Module, layer_name: str, device: torch.device = None):
        """
        Initialize Grad-CAM explainer.

        Args:
            model: The model to explain
            layer_name: Name of the convolutional layer to use for Grad-CAM
            device: Device to run on
        """
        super().__init__(model, device)
        self.layer_name = layer_name
        self.grad_cam = LayerGradCam(self.model, self._get_layer_by_name(layer_name))
        self.guided_grad_cam = GuidedGradCam(self.model, self._get_layer_by_name(layer_name))

    def _get_layer_by_name(self, layer_name: str):
        """Get layer by name from the model."""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found in model")

    def explain(self, input_tensor: torch.Tensor, target_class: Optional[int] = None,
                method: str = 'grad_cam') -> np.ndarray:
        """
        Generate Grad-CAM explanation.

        Args:
            input_tensor: Input image tensor
            target_class: Target class for explanation
            method: 'grad_cam' or 'guided_grad_cam'

        Returns:
            Attribution map as numpy array
        """
        input_tensor = self.preprocess_input(input_tensor)

        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()

        if method == 'grad_cam':
            attribution = self.grad_cam.attribute(input_tensor, target=target_class)
        elif method == 'guided_grad_cam':
            attribution = self.guided_grad_cam.attribute(input_tensor, target=target_class)
        else:
            raise ValueError("Method must be 'grad_cam' or 'guided_grad_cam'")

        return self.postprocess_attribution(attribution)


class GradCAMPlusPlus(ImageExplainerBase):
    """Grad-CAM++ implementation for improved localization."""

    def __init__(self, model: nn.Module, layer_name: str, device: torch.device = None):
        super().__init__(model, device)
        self.layer_name = layer_name
        self.target_layer = self._get_layer_by_name(layer_name)
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _get_layer_by_name(self, layer_name: str):
        """Get layer by name from the model."""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found in model")

    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def explain(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM++ explanation."""
        input_tensor = self.preprocess_input(input_tensor)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        class_score = output[:, target_class]
        class_score.backward()

        # Compute Grad-CAM++
        gradients = self.gradients
        activations = self.activations

        alpha = gradients.pow(2) / (2 * gradients.pow(2) +
                                   activations.mul(gradients.pow(3)).sum(dim=(2, 3), keepdim=True))
        alpha = torch.where(gradients != 0, alpha, torch.zeros_like(alpha))

        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam / (cam.max() + 1e-8)

        return self.postprocess_attribution(cam)


class SmoothGradExplainer(ImageExplainerBase):
    """SmoothGrad for reducing noise in gradients."""

    def __init__(self, model: nn.Module, device: torch.device = None,
                 noise_level: float = 0.15, n_samples: int = 50):
        super().__init__(model, device)
        self.noise_level = noise_level
        self.n_samples = n_samples
        self.saliency = Saliency(self.model)
        self.noise_tunnel = NoiseTunnel(self.saliency)

    def explain(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generate SmoothGrad explanation."""
        input_tensor = self.preprocess_input(input_tensor)

        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()

        attribution = self.noise_tunnel.attribute(
            input_tensor,
            target=target_class,
            stdevs=self.noise_level,
            n_samples=self.n_samples,
            nt_type='smoothgrad'
        )

        return self.postprocess_attribution(attribution)


class IntegratedGradientsExplainer(ImageExplainerBase):
    """Integrated Gradients explainability method."""

    def __init__(self, model: nn.Module, device: torch.device = None, n_steps: int = 50):
        super().__init__(model, device)
        self.n_steps = n_steps
        self.integrated_gradients = IntegratedGradients(self.model)

    def explain(self, input_tensor: torch.Tensor, target_class: Optional[int] = None,
                baseline: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Generate Integrated Gradients explanation.

        Args:
            input_tensor: Input image tensor
            target_class: Target class for explanation
            baseline: Baseline for comparison (default: zeros)

        Returns:
            Attribution map as numpy array
        """
        input_tensor = self.preprocess_input(input_tensor)

        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()

        attribution = self.integrated_gradients.attribute(
            input_tensor,
            baselines=baseline,
            target=target_class,
            n_steps=self.n_steps
        )

        return self.postprocess_attribution(attribution)


class ImageExplainer:
    """
    Comprehensive image explainer that combines multiple explanation methods.
    """

    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Initialize comprehensive image explainer.

        Args:
            model: Trained model to explain
            device: Device to run computations on
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.explainers = {}

        # Initialize available explainers
        self.explainers['saliency'] = VanillaSaliency(model, device)
        self.explainers['guided_backprop'] = GuidedBackpropagation(model, device)
        self.explainers['deeplift'] = DeepLIFTExplainer(model, device)
        self.explainers['integrated_gradients'] = IntegratedGradientsExplainer(model, device)
        self.explainers['smoothgrad'] = SmoothGradExplainer(model, device)

    def add_gradcam_explainer(self, layer_name: str):
        """Add Grad-CAM explainer for a specific layer."""
        try:
            self.explainers['gradcam'] = GradCAMExplainer(self.model, layer_name, self.device)
            self.explainers['gradcam_plus'] = GradCAMPlusPlus(self.model, layer_name, self.device)
        except Exception as e:
            logger.warning(f"Could not initialize Grad-CAM explainers: {e}")

    def explain(self, input_tensor: torch.Tensor, target_class: Optional[int] = None,
                methods: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Generate explanations using multiple methods.

        Args:
            input_tensor: Input image tensor
            target_class: Target class for explanation
            methods: List of methods to use (default: all available)

        Returns:
            Dictionary of attribution maps from different methods
        """
        if methods is None:
            methods = list(self.explainers.keys())

        explanations = {}

        for method in methods:
            if method not in self.explainers:
                logger.warning(f"Method {method} not available")
                continue

            try:
                attribution = self.explainers[method].explain(input_tensor, target_class)
                explanations[method] = attribution
            except Exception as e:
                logger.error(f"Error explaining with {method}: {e}")

        return explanations

    def visualize_explanations(self, input_tensor: torch.Tensor, target_class: Optional[int] = None,
                              methods: Optional[List[str]] = None, save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """Generate and save explanation visualizations."""
        explanations = self.explain(input_tensor, target_class, methods)
        figures = {}

        # Convert tensor to numpy for visualization
        if isinstance(input_tensor, torch.Tensor):
            if input_tensor.dim() == 4:
                input_image = input_tensor.squeeze(0).detach().cpu().numpy()
            else:
                input_image = input_tensor.detach().cpu().numpy()
        else:
            input_image = input_tensor

        for method, attribution in explanations.items():
            try:
                fig = visualize_attribution(
                    input_image, attribution, method,
                    save_path=f"{save_dir}/{method}_explanation.png" if save_dir else None
                )
                figures[method] = fig
            except Exception as e:
                logger.error(f"Error visualizing {method}: {e}")

        return figures


def visualize_attribution(image: np.ndarray, attribution: np.ndarray,
                         method_name: str, save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize attribution map overlaid on original image.

    Args:
        image: Original image as numpy array
        attribution: Attribution map as numpy array
        method_name: Name of the explanation method
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    if image.shape[0] == 3:  # CHW format
        image_show = np.transpose(image, (1, 2, 0))
    else:
        image_show = image

    axes[0].imshow(image_show)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Attribution map
    im1 = axes[1].imshow(attribution, cmap='RdYlBu_r')
    axes[1].set_title(f'{method_name} Attribution')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])

    # Overlay
    axes[2].imshow(image_show)
    axes[2].imshow(attribution, cmap='RdYlBu_r', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig