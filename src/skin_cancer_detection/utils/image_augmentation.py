"""
Utility script for image augmentation specialized for fluorescent microscopy images.

This module provides functions to:
1. Apply augmentations consistently across all 3 channels
2. Save augmentation parameters as metadata in filenames
3. Use appropriate augmentations for fluorescent microscopy data
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import random
from datetime import datetime
import math

logger = logging.getLogger(__name__)


class FluorescentMicroscopyAugmenter:
    """
    Augmentation class specialized for fluorescent microscopy images.
    Applies transformations consistently across all channels.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize augmenter.

        Args:
            seed: Random seed for reproducible augmentations
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.augmentation_functions = {
            'rotation': self._rotate,
            'horizontal_flip': self._horizontal_flip,
            'vertical_flip': self._vertical_flip,
            'brightness': self._adjust_brightness,
            'contrast': self._adjust_contrast,
            'noise': self._add_noise,
            'blur': self._add_blur,
            'crop': self._random_crop,
            'zoom': self._zoom,
            'shear': self._shear
        }

    def _rotate(self, image: Image.Image, angle: float) -> Image.Image:
        """Rotate image by specified angle."""
        return image.rotate(angle, expand=False, fillcolor=0)

    def _horizontal_flip(self, image: Image.Image, apply: bool) -> Image.Image:
        """Apply horizontal flip if flag is True."""
        if apply:
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image

    def _vertical_flip(self, image: Image.Image, apply: bool) -> Image.Image:
        """Apply vertical flip if flag is True."""
        if apply:
            return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return image

    def _adjust_brightness(self, image: Image.Image, factor: float) -> Image.Image:
        """Adjust brightness by factor."""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def _adjust_contrast(self, image: Image.Image, factor: float) -> Image.Image:
        """Adjust contrast by factor."""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def _add_noise(self, image: Image.Image, noise_factor: float) -> Image.Image:
        """Add Gaussian noise to image."""
        img_array = np.array(image)
        noise = np.random.normal(0, noise_factor * 255, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)

    def _add_blur(self, image: Image.Image, radius: float) -> Image.Image:
        """Apply Gaussian blur."""
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def _random_crop(self, image: Image.Image, crop_ratio: float) -> Image.Image:
        """Apply random crop and resize back to original size."""
        width, height = image.size
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)

        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        right = left + crop_width
        bottom = top + crop_height

        cropped = image.crop((left, top, right, bottom))
        return cropped.resize((width, height), Image.Resampling.LANCZOS)

    def _zoom(self, image: Image.Image, zoom_factor: float) -> Image.Image:
        """Apply zoom (scale) transformation."""
        width, height = image.size
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)

        # Resize image
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        if zoom_factor > 1.0:
            # Crop to original size from center
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            right = left + width
            bottom = top + height
            return resized.crop((left, top, right, bottom))
        else:
            # Pad to original size
            pad_width = (width - new_width) // 2
            pad_height = (height - new_height) // 2
            return ImageOps.expand(resized,
                                 border=(pad_width, pad_height,
                                        width - new_width - pad_width,
                                        height - new_height - pad_height),
                                 fill=0)

    def _shear(self, image: Image.Image, shear_factor: float) -> Image.Image:
        """Apply shear transformation."""
        # Convert to numpy for shear transformation
        img_array = np.array(image)

        # Create shear matrix
        shear_matrix = np.array([[1, shear_factor, 0],
                                [0, 1, 0]])

        # Apply transformation (simplified - for production use cv2 or skimage)
        # This is a basic implementation
        return image  # Placeholder - implement with proper library

    def generate_augmentation_params(self, augmentation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate random augmentation parameters based on configuration.

        Args:
            augmentation_config: Configuration for augmentation ranges

        Returns:
            Dictionary of augmentation parameters
        """
        params = {}

        # Rotation
        if 'rotation' in augmentation_config:
            angle_range = augmentation_config['rotation'].get('angle_range', [-15, 15])
            params['rotation'] = random.uniform(angle_range[0], angle_range[1])

        # Flips
        if 'horizontal_flip' in augmentation_config:
            prob = augmentation_config['horizontal_flip'].get('probability', 0.5)
            params['horizontal_flip'] = random.random() < prob

        if 'vertical_flip' in augmentation_config:
            prob = augmentation_config['vertical_flip'].get('probability', 0.5)
            params['vertical_flip'] = random.random() < prob

        # Brightness
        if 'brightness' in augmentation_config:
            range_val = augmentation_config['brightness'].get('factor_range', [0.8, 1.2])
            params['brightness'] = random.uniform(range_val[0], range_val[1])

        # Contrast
        if 'contrast' in augmentation_config:
            range_val = augmentation_config['contrast'].get('factor_range', [0.8, 1.2])
            params['contrast'] = random.uniform(range_val[0], range_val[1])

        # Noise
        if 'noise' in augmentation_config:
            range_val = augmentation_config['noise'].get('factor_range', [0.0, 0.05])
            params['noise'] = random.uniform(range_val[0], range_val[1])

        # Blur
        if 'blur' in augmentation_config:
            range_val = augmentation_config['blur'].get('radius_range', [0.0, 1.0])
            params['blur'] = random.uniform(range_val[0], range_val[1])

        # Crop
        if 'crop' in augmentation_config:
            range_val = augmentation_config['crop'].get('ratio_range', [0.8, 1.0])
            params['crop'] = random.uniform(range_val[0], range_val[1])

        # Zoom
        if 'zoom' in augmentation_config:
            range_val = augmentation_config['zoom'].get('factor_range', [0.9, 1.1])
            params['zoom'] = random.uniform(range_val[0], range_val[1])

        return params

    def apply_augmentations(self, image: Image.Image,
                           augmentation_params: Dict[str, Any]) -> Image.Image:
        """
        Apply augmentations to image using provided parameters.

        Args:
            image: PIL Image to augment
            augmentation_params: Dictionary of augmentation parameters

        Returns:
            Augmented PIL Image
        """
        augmented_image = image.copy()

        # Apply augmentations in a specific order
        augmentation_order = [
            'rotation', 'horizontal_flip', 'vertical_flip',
            'crop', 'zoom', 'brightness', 'contrast', 'noise', 'blur'
        ]

        for aug_type in augmentation_order:
            if aug_type in augmentation_params:
                param_value = augmentation_params[aug_type]
                if aug_type in self.augmentation_functions:
                    augmented_image = self.augmentation_functions[aug_type](
                        augmented_image, param_value)

        return augmented_image


def create_default_augmentation_config() -> Dict[str, Any]:
    """
    Create default augmentation configuration for fluorescent microscopy images.

    Returns:
        Default augmentation configuration
    """
    return {
        'rotation': {'angle_range': [-10, 10]},
        'horizontal_flip': {'probability': 0.5},
        'vertical_flip': {'probability': 0.5},
        'brightness': {'factor_range': [0.9, 1.1]},
        'contrast': {'factor_range': [0.9, 1.1]},
        'noise': {'factor_range': [0.0, 0.02]},  # Low noise for microscopy
        'blur': {'radius_range': [0.0, 0.5]},   # Minimal blur
        # 'crop': {'ratio_range': [0.9, 1.0]},    # Commented out - changes image size
        # 'zoom': {'factor_range': [0.95, 1.05]}  # Commented out - changes image size
    }


def encode_augmentation_in_filename(original_filename: str,
                                   augmentation_params: Dict[str, Any],
                                   augmentation_id: str,
                                   simple_naming: bool = True) -> str:
    """
    Encode augmentation parameters in filename using dash separator.

    Args:
        original_filename: Original filename
        augmentation_params: Applied augmentation parameters
        augmentation_id: Unique identifier for this augmentation
        simple_naming: If True, use only aug_id; if False, include all parameters

    Returns:
        New filename with augmentation metadata
    """
    path = Path(original_filename)
    stem = path.stem
    suffix = path.suffix

    if simple_naming:
        # Simple naming: original-aug-001.ext
        new_filename = f"{stem}-aug-{augmentation_id}{suffix}"
    else:
        # Detailed naming with parameters
        param_parts = []
        for key, value in augmentation_params.items():
            if isinstance(value, bool):
                if value:
                    param_parts.append(f"{key[:3]}")  # Abbreviated parameter name
            elif isinstance(value, (int, float)):
                param_parts.append(f"{key[:3]}{value:.2f}".replace('.', 'p'))

        param_string = "-".join(param_parts)[:50]  # Limit length, use dashes
        new_filename = f"{stem}-aug-{augmentation_id}-{param_string}{suffix}"

    return new_filename


def process_image_augmentation(input_dir: Path, output_dir: Path,
                              augmentation_config: Optional[Dict[str, Any]] = None,
                              num_augmentations_per_image: int = 3,
                              simple_naming: bool = True,
                              save_original: bool = True,
                              seed: Optional[int] = None) -> Dict:
    """
    Process images by applying augmentations and saving with metadata.

    Args:
        input_dir: Input directory with images
        output_dir: Output directory for augmented images
        augmentation_config: Augmentation configuration
        num_augmentations_per_image: Number of augmented versions per image
        simple_naming: If True, use simple aug-id naming; if False, include parameters
        save_original: If True, save original image with aug-000 id
        seed: Random seed for reproducibility

    Returns:
        Dictionary with processing statistics and metadata
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use default config if none provided
    if augmentation_config is None:
        augmentation_config = create_default_augmentation_config()

    # Initialize augmenter
    augmenter = FluorescentMicroscopyAugmenter(seed=seed)

    # Process images
    processed_count = 0
    error_count = 0
    augmented_count = 0
    original_saved_count = 0
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}

    augmentation_metadata = {
        "processing_info": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "num_augmentations_per_image": num_augmentations_per_image,
            "simple_naming": simple_naming,
            "save_original": save_original,
            "seed": seed
        },
        "augmentation_config": augmentation_config,
        "augmented_files": []
    }

    for file_path in input_dir.rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            try:
                # Skip if already augmented (contains "-aug-")
                if "-aug-" in file_path.stem:
                    continue

                processed_count += 1

                # Load original image
                with Image.open(file_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Maintain directory structure
                    relative_path = file_path.relative_to(input_dir)
                    output_subdir = output_dir / relative_path.parent
                    output_subdir.mkdir(parents=True, exist_ok=True)

                    # Save original image with aug-000 if requested
                    if save_original:
                        original_filename = encode_augmentation_in_filename(
                            file_path.name, {}, "000", simple_naming)
                        original_output_path = output_subdir / original_filename

                        img.save(original_output_path, quality=95)
                        original_saved_count += 1

                        # Record metadata for original
                        original_metadata = {
                            "original_file": str(file_path),
                            "augmented_file": str(original_output_path),
                            "augmentation_id": "000",
                            "augmentation_params": {},  # No augmentation for original
                            "original_size": img.size,
                            "augmented_size": img.size
                        }
                        augmentation_metadata["augmented_files"].append(original_metadata)

                    # Generate multiple augmented versions
                    for aug_idx in range(num_augmentations_per_image):
                        # Generate augmentation parameters
                        aug_params = augmenter.generate_augmentation_params(
                            augmentation_config)

                        # Apply augmentations
                        augmented_img = augmenter.apply_augmentations(img, aug_params)

                        # Create augmentation ID (starting from 001 since 000 is for original)
                        aug_id = f"{aug_idx + 1:03d}"

                        # Create filename with augmentation metadata
                        new_filename = encode_augmentation_in_filename(
                            file_path.name, aug_params, aug_id, simple_naming)

                        output_file_path = output_subdir / new_filename

                        # Save augmented image
                        augmented_img.save(output_file_path, quality=95)

                        # Record metadata
                        file_metadata = {
                            "original_file": str(file_path),
                            "augmented_file": str(output_file_path),
                            "augmentation_id": aug_id,
                            "augmentation_params": aug_params,
                            "original_size": img.size,
                            "augmented_size": augmented_img.size
                        }
                        augmentation_metadata["augmented_files"].append(file_metadata)

                        augmented_count += 1

                if processed_count % 10 == 0:
                    logger.info(f"Processed {processed_count} original images, "
                              f"saved {original_saved_count} originals, "
                              f"created {augmented_count} augmented versions...")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                error_count += 1

    # Save metadata
    metadata_file = output_dir / "augmentation_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(augmentation_metadata, f, indent=2)

    # Save augmentation config for reference
    config_file = output_dir / "augmentation_config.json"
    with open(config_file, 'w') as f:
        json.dump(augmentation_config, f, indent=2)

    stats = {
        "processed_original_count": processed_count,
        "original_saved_count": original_saved_count,
        "augmented_count": augmented_count,
        "total_output_count": original_saved_count + augmented_count,
        "error_count": error_count,
        "augmentations_per_image": num_augmentations_per_image,
        "save_original": save_original,
        "metadata_file": str(metadata_file),
        "config_file": str(config_file)
    }

    logger.info(f"Augmentation complete. Processed: {processed_count} originals, "
              f"Saved originals: {original_saved_count}, "
              f"Created: {augmented_count} augmented, Errors: {error_count}")

    return stats


if __name__ == "__main__":
    # Example usage
    input_directory = Path("data/03_primary")
    output_directory = Path("data/04_feature")

    # Custom augmentation config for fluorescent microscopy
    custom_config = create_default_augmentation_config()

    # Process images with augmentation
    stats = process_image_augmentation(
        input_dir=input_directory,
        output_dir=output_directory,
        augmentation_config=custom_config,
        num_augmentations_per_image=5,
        simple_naming=True,
        save_original=True,
        seed=42
    )

    print(f"Augmentation completed: {stats}")