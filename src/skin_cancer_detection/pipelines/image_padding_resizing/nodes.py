"""
Nodes for image padding and resizing pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, Any

from skin_cancer_detection.utils.padding_and_resizing import (
    process_image_directory,
    calculate_padding_size
)

logger = logging.getLogger(__name__)


def pad_and_resize_images(input_dir: str, output_dir: str,
                         resize_for_model: str = "resnet18",
                         target_padding_size: tuple = None,
                         force_square: bool = True) -> Dict[str, Any]:
    """
    Pad and resize images for CNN training.

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        resize_for_model: CNN model type for final resizing
        target_padding_size: Target size for padding (width, height)
        force_square: If True, pad images to square dimensions to avoid deformation

    Returns:
        Dictionary with processing statistics
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    logger.info(f"Starting image padding and resizing from {input_path} to {output_path}")
    logger.info(f"Force square padding: {force_square}")

    # Process images
    stats = process_image_directory(
        input_dir=input_path,
        output_dir=output_path,
        target_padding_size=target_padding_size,
        resize_for_model=resize_for_model,
        force_square=force_square
    )

    logger.info(f"Completed image padding and resizing: {stats}")

    return stats


def calculate_optimal_padding_size(input_dir: str, force_square: bool = True) -> Dict[str, Any]:
    """
    Calculate optimal padding size for images in directory.

    Args:
        input_dir: Input directory path
        force_square: If True, calculate square dimensions to avoid deformation

    Returns:
        Dictionary with calculated dimensions
    """
    input_path = Path(input_dir)

    logger.info(f"Calculating optimal padding size for {input_path}")
    logger.info(f"Force square padding: {force_square}")

    max_width, max_height = calculate_padding_size(input_path, force_square=force_square)

    result = {
        "max_width": max_width,
        "max_height": max_height,
        "input_dir": str(input_path),
        "force_square": force_square
    }

    logger.info(f"Calculated padding size: {max_width}x{max_height}")

    return result