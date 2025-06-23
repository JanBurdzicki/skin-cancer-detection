"""
Explainable AI (XAI) module for skin cancer detection.

This module provides comprehensive explainability methods for both image and tabular data.
"""

from .image_explainer import (
    VanillaSaliency,
    GuidedBackpropagation,
    DeepLIFTExplainer,
    GradCAMExplainer,
    GradCAMPlusPlus,
    SmoothGradExplainer,
    IntegratedGradientsExplainer,
    ImageExplainerBase
)

from .tabular_explainer import (
    SHAPExplainer,
    PermutationImportanceExplainer,
    LIMEExplainer,
    PDPExplainer,
    ICEExplainer,
    FeatureImportanceExplainer,
    CoefficientsExplainer,
    TreeInterpreterExplainer,
    TabularExplainerBase
)

from .utils import (
    save_explanation,
    load_explanation,
    plot_explanations,
    ExplanationConfig
)

__all__ = [
    # Image explainers
    "VanillaSaliency",
    "GuidedBackpropagation", 
    "DeepLIFTExplainer",
    "GradCAMExplainer",
    "GradCAMPlusPlus",
    "SmoothGradExplainer",
    "IntegratedGradientsExplainer",
    "ImageExplainerBase",
    
    # Tabular explainers
    "SHAPExplainer",
    "PermutationImportanceExplainer", 
    "LIMEExplainer",
    "PDPExplainer",
    "ICEExplainer",
    "FeatureImportanceExplainer",
    "CoefficientsExplainer",
    "TreeInterpreterExplainer",
    "TabularExplainerBase",
    
    # Utilities
    "save_explanation",
    "load_explanation", 
    "plot_explanations",
    "ExplanationConfig"
] 