"""
Nodes for image augmentation pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, Any

from skin_cancer_detection.utils.image_augmentation import (
    process_image_augmentation,
    create_default_augmentation_config,
    FluorescentMicroscopyAugmenter
)

logger = logging.getLogger(__name__)


def augment_images_for_training(input_dir: str, output_dir: str,
                               num_augmentations_per_image: int = 3,
                               augmentation_config: Dict[str, Any] = None,
                               simple_naming: bool = True,
                               save_original: bool = True,
                               seed: int = 42) -> Dict[str, Any]:
    """
    Apply augmentations to images for training data expansion.

    Args:
        input_dir: Input directory with images
        output_dir: Output directory for augmented images
        num_augmentations_per_image: Number of augmented versions per image
        augmentation_config: Configuration for augmentation parameters
        simple_naming: If True, use simple aug-id naming; if False, include parameters
        save_original: If True, save original image with aug-000 id
        seed: Random seed for reproducibility

    Returns:
        Dictionary with processing statistics
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    logger.info(f"Starting image augmentation from {input_path} to {output_path}")
    logger.info(f"Creating {num_augmentations_per_image} augmented versions per image")
    logger.info(f"Simple naming: {simple_naming}")
    logger.info(f"Save original: {save_original}")

    # Use default config if none provided
    if augmentation_config is None:
        augmentation_config = create_default_augmentation_config()

    # Process image augmentation
    stats = process_image_augmentation(
        input_dir=input_path,
        output_dir=output_path,
        augmentation_config=augmentation_config,
        num_augmentations_per_image=num_augmentations_per_image,
        simple_naming=simple_naming,
        save_original=save_original,
        seed=seed
    )

    logger.info(f"Completed image augmentation: {stats}")

    return stats


def create_fluorescent_microscopy_augmentation_config() -> Dict[str, Any]:
    """
    Create augmentation configuration optimized for fluorescent microscopy images.

    Returns:
        Dictionary with augmentation configuration
    """
    logger.info("Creating fluorescent microscopy augmentation configuration")

    # Create specialized config for fluorescent microscopy
    # Crop and zoom are commented out to maintain image size
    config = {
        'rotation': {'angle_range': [-15, 15]},  # Moderate rotation
        'horizontal_flip': {'probability': 0.5},
        'vertical_flip': {'probability': 0.5},
        'brightness': {'factor_range': [0.85, 1.15]},  # Moderate brightness changes
        'contrast': {'factor_range': [0.9, 1.1]},      # Minimal contrast changes
        'noise': {'factor_range': [0.0, 0.01]},        # Very low noise
        'blur': {'radius_range': [0.0, 0.3]},          # Minimal blur
        # 'crop': {'ratio_range': [0.95, 1.0]},          # Commented out - changes image size
        # 'zoom': {'factor_range': [0.98, 1.02]}         # Commented out - changes image size
    }

    logger.info(f"Created augmentation config: {config}")

    return config


def validate_augmentation_results(augmentation_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the results of image augmentation.

    Args:
        augmentation_stats: Statistics from augmentation process

    Returns:
        Dictionary with validation results
    """
    logger.info("Validating augmentation results")

    processed_count = augmentation_stats.get('processed_original_count', 0)
    augmented_count = augmentation_stats.get('augmented_count', 0)
    error_count = augmentation_stats.get('error_count', 0)
    augmentations_per_image = augmentation_stats.get('augmentations_per_image', 0)

    # Expected augmented count
    expected_augmented = processed_count * augmentations_per_image

    # Calculate success rate
    success_rate = (augmented_count / expected_augmented) * 100 if expected_augmented > 0 else 0

    validation_results = {
        "processed_original_images": processed_count,
        "total_augmented_images": augmented_count,
        "expected_augmented_images": expected_augmented,
        "error_count": error_count,
        "success_rate_percent": success_rate,
        "augmentations_per_image": augmentations_per_image,
        "augmentation_factor": (augmented_count + processed_count) / processed_count if processed_count > 0 else 0,
        "validation_status": "PASS" if success_rate >= 95 else "FAIL"
    }

    logger.info(f"Augmentation validation results: {validation_results}")

    return validation_results


def analyze_augmentation_parameters(augmentation_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the distribution of augmentation parameters used.

    Args:
        augmentation_stats: Statistics from augmentation process

    Returns:
        Dictionary with parameter analysis
    """
    logger.info("Analyzing augmentation parameters")

    # This would typically load the augmentation metadata file
    # For now, we'll return a placeholder analysis

    analysis = {
        "total_augmentations": augmentation_stats.get('augmented_count', 0),
        "parameter_distribution": {
            "rotation": {"min": -15, "max": 15, "avg": 0},
            "brightness": {"min": 0.85, "max": 1.15, "avg": 1.0},
            "contrast": {"min": 0.9, "max": 1.1, "avg": 1.0}
        },
        # Crop and zoom are commented out to maintain image size
        "augmentation_types_used": [
            "rotation", "horizontal_flip", "vertical_flip",
            "brightness", "contrast", "noise", "blur" # , "crop", "zoom"
        ]
    }

    logger.info("Completed augmentation parameter analysis")

    return analysis