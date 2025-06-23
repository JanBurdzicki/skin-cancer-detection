"""
Utility script for splitting data into train, test, and validation sets.

This module provides functions to:
1. Split tabular data using stratified sampling for imbalanced datasets
2. Match and organize corresponding image data
3. Save splits in organized directory structure for different model types
4. Handle both tabular features and image data consistently
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re

logger = logging.getLogger(__name__)


def load_feature_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load feature data from parquet file.

    Args:
        file_path: Path to the parquet file

    Returns:
        DataFrame with feature data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Feature data file not found: {file_path}")

    logger.info(f"Loading feature data from: {file_path}")
    df = pd.read_parquet(file_path)

    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")

    return df


def stratified_split_data(
    df: pd.DataFrame,
    label_col: str = 'gfp_status',
    sample_col: str = 'sample_name',
    roi_col: str = 'roi_name',
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets using stratified sampling.
    Preserves sample_name and roi_name columns for cell identification.

    Args:
        df: Input DataFrame
        label_col: Column name for labels
        sample_col: Sample identifier column (preserved)
        roi_col: ROI identifier column (preserved)
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining data)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info(f"Splitting data with stratification on '{label_col}'")
    logger.info(f"Preserving columns: {sample_col}, {roi_col}")
    logger.info(f"Split sizes: test={test_size}, val={val_size}")

    # Check if required columns exist
    required_cols = [label_col, sample_col, roi_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Log class distribution
    class_counts = df[label_col].value_counts().sort_index()
    logger.info(f"Original class distribution: {class_counts.to_dict()}")

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state
    )

    # Second split: separate validation from training
    # Adjust val_size to be relative to remaining data
    adjusted_val_size = val_size / (1 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        stratify=train_val_df[label_col],
        random_state=random_state
    )

    # Log final distributions
    logger.info(f"Final split sizes:")
    logger.info(f"  Train: {len(train_df)} samples ({len(train_df)/len(df):.3f})")
    logger.info(f"  Val: {len(val_df)} samples ({len(val_df)/len(df):.3f})")
    logger.info(f"  Test: {len(test_df)} samples ({len(test_df)/len(df):.3f})")

    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        split_counts = split_df[label_col].value_counts().sort_index()
        logger.info(f"  {split_name} class distribution: {split_counts.to_dict()}")

    # Verify all required columns are preserved
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for col in [sample_col, roi_col]:
            if col not in split_df.columns:
                logger.warning(f"Column '{col}' missing from {split_name} split")

    return train_df, val_df, test_df


def extract_sample_roi_label_from_filename(filename: str) -> Tuple[str, str, int]:
    """
    Extract sample_name, roi_name, and label from image filename using regex.
    Handles augmented images with numbering (aug-000, aug-001, etc.).

    Expected formats:
    - cell-r03c01-f01-ch03-ID0004-negative-aug-000.tiff
    - cell-r03c01-f01-ch03-ID0004-positive-aug-001.tiff
    - cell-r03c01-f01-ch03-ID0004-negative.tiff (without aug)

    Returns: ('r03c01-f01', 'ID0004', 0)  # 0 for negative, 1 for positive

    Args:
        filename: Image filename

    Returns:
        Tuple of (sample_name, roi_name, label) or (None, None, None) if parsing fails
    """
    try:
        # Remove extension
        name = Path(filename).stem

        # Primary regex pattern - handles both with and without aug-id
        # Pattern matches: cell-r03c01-f01-ch03-ID0004-negative[-aug-000]
        pattern = r'cell-(r\d+c\d+-f\d+)-ch\d+-(ID\d+)-(negative|positive)(?:-aug-\d+)?'
        match = re.search(pattern, name, re.IGNORECASE)

        if match:
            sample_name = match.group(1)  # e.g., 'r03c01-f01'
            roi_name = match.group(2)     # e.g., 'ID0004'
            label_str = match.group(3)    # 'negative' or 'positive'

            # Convert label to integer
            label = 0 if label_str == 'negative' else 1

            return sample_name, roi_name, label
        else:
            logger.debug(f"Regex pattern did not match filename: {filename}")
            return None, None, None

    except Exception as e:
        logger.debug(f"Failed to parse filename {filename}: {e}")
        return None, None, None


def find_images_for_split(
    split_df: pd.DataFrame,
    image_dir: Path,
    sample_col: str = 'sample_name',
    roi_col: str = 'roi_name',
    channel_filter: str = 'ch03'
) -> List[Path]:
    """
    Find all ch03 images in patches/mt directories that belong to the current split.

    Args:
        split_df: DataFrame for current split (train/val/test)
        image_dir: Base directory containing image files
        sample_col: Column name for sample identifier
        roi_col: Column name for ROI identifier
        channel_filter: Channel identifier to filter by (e.g., 'ch03')

    Returns:
        List of image paths that belong to this split
    """
    if not image_dir.exists():
        logger.warning(f"Image directory does not exist: {image_dir}")
        return []

    # Create lookup set for fast membership testing
    split_keys = set()
    for _, row in split_df.iterrows():
        sample_name = str(row[sample_col])
        roi_name = str(row[roi_col])
        split_keys.add((sample_name, roi_name))

    unique_combinations = len(split_keys)
    dataset_samples_count = len(split_df)

    logger.info(f"Looking for images from {unique_combinations} unique sample-roi combinations")
    logger.info(f"Scanning patches/mt directories for ch03 images")

    # Find all ch03 images in patches/mt directories
    matching_images = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}

    # Scan for patches/mt directories
    for patches_dir in image_dir.rglob('patches'):
        mt_dir = patches_dir / 'mt'
        if mt_dir.exists() and mt_dir.is_dir():
            logger.debug(f"Scanning MT directory: {mt_dir}")

            for img_path in mt_dir.rglob('*'):
                if img_path.suffix.lower() in image_extensions:
                    img_name = img_path.name

                    # Check if filename contains ch03
                    if channel_filter.lower() in img_name.lower():
                        # Extract sample_name and roi_name from filename
                        sample_name, roi_name, _ = extract_sample_roi_label_from_filename(img_name)

                        if sample_name and roi_name:
                            # Check if this image belongs to current split
                            if (sample_name, roi_name) in split_keys:
                                matching_images.append(img_path)
                                logger.debug(f"Found matching image: {img_name} -> {sample_name}, {roi_name}")

    # Log results with improved messaging
    all_files_also_augmented_count = len(matching_images)
    logger.info(f"Found {all_files_also_augmented_count} ch03 images for {dataset_samples_count} dataset samples")

    return matching_images


def save_tabular_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    label_col: str = 'gfp_status'
) -> Dict[str, str]:
    """
    Save train/validation/test splits as parquet files in tabular/ subdirectory.
    Everything except label_col is treated as features.

    Args:
        train_df, val_df, test_df: Split DataFrames
        output_dir: Output directory
        label_col: Label column name

    Returns:
        Dictionary with saved file paths
    """
    # Create tabular subdirectory
    tabular_dir = output_dir / 'tabular'
    tabular_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

    saved_files = {}

    for split_name, df in splits.items():
        # Save complete dataset in tabular subdirectory
        filename = f"{split_name}.parquet"
        filepath = tabular_dir / filename

        df.to_parquet(filepath, index=False)
        saved_files[split_name] = str(filepath)

        logger.info(f"Saved {split_name} split: {len(df)} samples to tabular/{filename}")

    return saved_files


def organize_image_splits(
    train_images: List[Path],
    val_images: List[Path],
    test_images: List[Path],
    output_dir: Path,
    channel_filter: str = 'ch03'
) -> None:
    """
    Organize images into images/train/val/test directories with class subdirectories.

    Args:
        train_images: List of training image paths
        val_images: List of validation image paths
        test_images: List of test image paths
        output_dir: Base output directory
        channel_filter: Channel identifier for logging
    """
    # Create images subdirectory
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Organizing {channel_filter} images into images/train/val/test directories")

    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

    for split_name, image_list in splits.items():
        logger.info(f"Processing {split_name} split: {len(image_list)} images")

        # Create split directory in images subdirectory
        split_dir = images_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Create class subdirectories
        for class_name in ['negative', 'positive']:
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

        # Copy images to appropriate class directories
        class_counts = {'negative': 0, 'positive': 0}

        for img_path in image_list:
            # Extract label from filename
            _, _, label = extract_sample_roi_label_from_filename(img_path.name)

            if label is not None:
                class_name = 'negative' if label == 0 else 'positive'
                class_counts[class_name] += 1

                # Destination path in images subdirectory
                dest_path = split_dir / class_name / img_path.name

                # Copy image
                try:
                    shutil.copy2(img_path, dest_path)
                    logger.debug(f"Copied {img_path.name} to images/{split_name}/{class_name}/")
                except Exception as e:
                    logger.error(f"Failed to copy {img_path}: {e}")
            else:
                logger.warning(f"Could not extract label from {img_path.name}")

        # Log class distribution for this split
        logger.info(f"  {split_name.capitalize()} class distribution: {class_counts}")

    logger.info("Image organization completed")


def process_data_splitting(
    input_file: str,
    output_dir: str,
    image_data_dir: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    label_col: str = 'gfp_status',
    sample_col: str = 'sample_name',
    roi_col: str = 'roi_name',
    channel_filter: str = 'ch03'
) -> Dict[str, Any]:
    """
    Complete data splitting pipeline.

    Args:
        input_file: Path to input parquet file
        output_dir: Output directory for splits
        image_data_dir: Directory containing image data
        test_size: Test set proportion
        val_size: Validation set proportion
        random_state: Random seed
        label_col: Label column name
        sample_col: Sample identifier column
        roi_col: ROI identifier column
        channel_filter: Image channel filter

    Returns:
        Dictionary with processing statistics
    """
    logger.info("Starting data splitting pipeline")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_dir}")

    # Convert paths
    input_path = Path(input_file)
    output_path = Path(output_dir)
    image_data_dir = Path(image_data_dir)

    # Load data
    logger.info("Loading feature data")
    df = load_feature_data(input_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Perform stratified split
    train_df, val_df, test_df = stratified_split_data(
        df, label_col, sample_col, roi_col, test_size, val_size, random_state
    )

    # Save tabular splits
    tabular_files = save_tabular_splits(
        train_df, val_df, test_df, output_path, label_col
    )

    # Organize image splits
    train_images = find_images_for_split(train_df, image_data_dir)
    val_images = find_images_for_split(val_df, image_data_dir)
    test_images = find_images_for_split(test_df, image_data_dir)

    organize_image_splits(train_images, val_images, test_images, output_path, channel_filter)

    # Calculate total images found
    total_images = len(train_images) + len(val_images) + len(test_images)

    # Compile statistics
    stats = {
        "input_file": str(input_path),
        "output_directory": str(output_path),
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "total": len(df)
        },
        "tabular_files": tabular_files,
        "image_organization": "by_split_and_class",
        "total_images_found": total_images,
        "config": {
            "test_size": test_size,
            "val_size": val_size,
            "random_state": random_state,
            "label_col": label_col,
            "sample_col": sample_col,
            "roi_col": roi_col,
            "channel_filter": channel_filter
        }
    }

    logger.info("Data splitting pipeline completed successfully")
    logger.info(f"Created {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
    logger.info(f"Organized {total_images} images total")

    return stats


if __name__ == "__main__":
    # Example usage
    feature_file = "data/04_feature/feature_data.parquet"
    image_dir = "data/04_feature"
    output_dir = "data/05_model_input"

    # Run data splitting
    stats = process_data_splitting(
        input_file=feature_file,
        output_dir=output_dir,
        image_data_dir=image_dir,
        test_size=0.2,
        val_size=0.2,
        random_state=42,
        label_col='gfp_status',
        sample_col='sample_name',
        roi_col='roi_name',
        channel_filter='ch03'
    )

    print("Data splitting completed!")
    print(f"Statistics: {stats}")