"""
Utility script for padding and resizing images for CNN training.

This module provides functions to:
1. Calculate padding dimensions to make all images uniform size
2. Apply padding to images
3. Resize images to target dimensions (e.g., for ResNet18: 224x224)
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
from PIL import Image, ImageOps
import json

logger = logging.getLogger(__name__)


def calculate_padding_size(image_dir: Path, target_size: Optional[Tuple[int, int]] = None,
                          force_square: bool = True) -> Tuple[int, int]:
    """
    Calculate the maximum dimensions across all images in a directory.

    Args:
        image_dir: Path to directory containing images
        target_size: Optional target size (width, height). If None, uses max dimensions found.
        force_square: If True, returns square dimensions (max_dimension, max_dimension)

    Returns:
        Tuple of (width, height) for padding calculations
    """
    if not image_dir.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")

    max_width = 0
    max_height = 0
    image_count = 0

    # Support common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}

    for file_path in image_dir.rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    max_width = max(max_width, width)
                    max_height = max(max_height, height)
                    image_count += 1
            except Exception as e:
                logger.warning(f"Could not process image {file_path}: {e}")

    if image_count == 0:
        raise ValueError(f"No valid images found in {image_dir}")

    logger.info(f"Processed {image_count} images. Max dimensions: {max_width}x{max_height}")

    # Use target size if provided
    if target_size:
        if force_square and target_size[0] != target_size[1]:
            # Make target square by using the larger dimension
            max_dim = max(target_size)
            logger.info(f"Forcing square dimensions: {max_dim}x{max_dim}")
            return (max_dim, max_dim)
        return target_size

    # Calculate padding size
    if force_square:
        # Use the larger of max_width and max_height to create square padding
        max_dimension = max(max_width, max_height)
        logger.info(f"Calculated square padding size: {max_dimension}x{max_dimension}")
        return (max_dimension, max_dimension)

    return (max_width, max_height)


def pad_image(image: Image.Image, target_size: Tuple[int, int],
              fill_color: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """
    Pad an image to target dimensions.

    Args:
        image: PIL Image to pad
        target_size: Target (width, height)
        fill_color: RGB color for padding (default: black)

    Returns:
        Padded PIL Image
    """
    current_width, current_height = image.size
    target_width, target_height = target_size

    if current_width > target_width or current_height > target_height:
        logger.warning(f"Image size ({current_width}x{current_height}) exceeds target size ({target_width}x{target_height})")

    # Calculate padding
    pad_width = max(0, target_width - current_width)
    pad_height = max(0, target_height - current_height)

    # Center the image by padding equally on both sides
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad

    # Apply padding
    padded_image = ImageOps.expand(
        image,
        border=(left_pad, top_pad, right_pad, bottom_pad),
        fill=fill_color
    )

    return padded_image


def resize_for_cnn(image: Image.Image, model_type: str = "resnet18") -> Image.Image:
    """
    Resize image to dimensions suitable for popular CNN architectures.

    Args:
        image: PIL Image to resize
        model_type: CNN architecture ("resnet18", "resnet50", "vgg16", "custom")

    Returns:
        Resized PIL Image
    """
    # Standard input sizes for popular architectures
    standard_sizes = {
        "resnet18": (224, 224),
        "resnet50": (224, 224),
        "vgg16": (224, 224),
        "inception": (299, 299),
        "custom": (256, 256)  # Default custom size
    }

    target_size = standard_sizes.get(model_type.lower(), (224, 224))

    # Use high-quality resampling for fluorescent microscopy images
    resized_image = image.resize(target_size, Image.Resampling.LANCZOS)

    return resized_image


def process_image_directory(input_dir: Path, output_dir: Path,
                          target_padding_size: Optional[Tuple[int, int]] = None,
                          resize_for_model: str = "resnet18",
                          fill_color: Tuple[int, int, int] = (0, 0, 0),
                          force_square: bool = True) -> dict:
    """
    Process all images in a directory: pad to uniform size, then resize for CNN.

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        target_padding_size: Target size for padding. If None, calculated automatically.
        resize_for_model: CNN model type for final resizing
        fill_color: RGB color for padding
        force_square: If True, pad images to square dimensions to avoid deformation

    Returns:
        Dictionary with processing statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate padding size if not provided
    if target_padding_size is None:
        target_padding_size = calculate_padding_size(input_dir, force_square=force_square)

    logger.info(f"Using padding size: {target_padding_size}")
    logger.info(f"Resizing for model: {resize_for_model}")

    # Process images
    processed_count = 0
    error_count = 0
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}

    # Create metadata file
    metadata = {
        "processing_params": {
            "target_padding_size": target_padding_size,
            "resize_for_model": resize_for_model,
            "fill_color": fill_color,
            "force_square": force_square
        },
        "processed_files": []
    }

    for file_path in input_dir.rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            try:
                # Maintain directory structure
                relative_path = file_path.relative_to(input_dir)
                output_file_path = output_dir / relative_path
                output_file_path.parent.mkdir(parents=True, exist_ok=True)

                # Process image
                with Image.open(file_path) as img:
                    # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    original_size = img.size

                    # Step 1: Pad to uniform size
                    padded_img = pad_image(img, target_padding_size, fill_color)

                    # Step 2: Resize for CNN
                    final_img = resize_for_cnn(padded_img, resize_for_model)

                    # Save processed image
                    final_img.save(output_file_path, quality=95)

                    # Record metadata
                    file_metadata = {
                        "original_file": str(file_path),
                        "output_file": str(output_file_path),
                        "original_size": original_size,
                        "padded_size": target_padding_size,
                        "final_size": final_img.size
                    }
                    metadata["processed_files"].append(file_metadata)

                    processed_count += 1

                    if processed_count % 100 == 0:
                        logger.info(f"Processed {processed_count} images...")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                error_count += 1

    # Save metadata
    metadata_file = output_dir / "padding_resizing_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    stats = {
        "processed_count": processed_count,
        "error_count": error_count,
        "target_padding_size": target_padding_size,
        "resize_for_model": resize_for_model,
        "metadata_file": str(metadata_file)
    }

    logger.info(f"Processing complete. Processed: {processed_count}, Errors: {error_count}")

    return stats


if __name__ == "__main__":
    # Example usage
    input_directory = Path("data/03_primary")
    output_directory = Path("data/03_primary")

    # Process images with default settings (ResNet18-compatible)
    stats = process_image_directory(
        input_dir=input_directory,
        output_dir=output_directory,
        resize_for_model="resnet18"
    )

    print(f"Processing completed: {stats}")