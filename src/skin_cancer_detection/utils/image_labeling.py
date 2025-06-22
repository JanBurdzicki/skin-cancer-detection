"""
Utility script for adding labels to images based on gfp_status.

This module provides functions to:
1. Extract gfp_status from parquet files using sample_name and roi_name
2. Add labels to filenames and save to output directory
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import shutil
import re

logger = logging.getLogger(__name__)


def load_label_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load label data from parquet/CSV file containing gfp_status information.

    Args:
        csv_path: Path to parquet/CSV file with gfp_status column

    Returns:
        DataFrame with label information
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise ValueError(f"File does not exist: {csv_path}")

    # Try different file formats
    if csv_path.suffix.lower() == '.parquet':
        df = pd.read_parquet(csv_path)
    elif csv_path.suffix.lower() == '.csv':
        df = pd.read_csv(csv_path)
    else:
        raise ValueError(f"Unsupported file format: {csv_path.suffix}")

    # Validate required columns
    required_columns = ['sample_name', 'roi_name', 'gfp_status']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {csv_path}")

    logger.info(f"Loaded label data with {len(df)} records from {csv_path}")

    # Log distribution of labels
    if 'gfp_status' in df.columns:
        label_counts = df['gfp_status'].value_counts()
        logger.info(f"Label distribution: {label_counts.to_dict()}")

    return df


def add_label_to_filename(original_filename: str, label: str) -> str:
    """
    Add label to filename simply.

    Args:
        original_filename: Original filename (e.g., cell-r01c01-f01-ch01-ID0005.tiff)
        label: String label (e.g., 'positive', 'negative')

    Returns:
        New filename with label (e.g., cell-r01c01-f01-ch01-ID0005-positive.tiff)
    """
    path = Path(original_filename)
    stem = path.stem
    suffix = path.suffix

    # Simple append: original_stem-label.extension
    new_filename = f"{stem}-{label}{suffix}"

    return new_filename


def match_images_to_labels(image_dir: Path, label_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Match image files to their corresponding labels using sample_name and roi_name.
    Extract sample_name and roi_name from filename using regex for efficient matching.

    Args:
        image_dir: Directory containing images
        label_df: DataFrame with label information (must have sample_name, roi_name, gfp_status)

    Returns:
        Dictionary mapping image paths to label information
    """
    required_columns = ['sample_name', 'roi_name', 'gfp_status']
    for col in required_columns:
        if col not in label_df.columns:
            raise ValueError(f"Required column '{col}' not found in label data")

    # Create simple lookup dictionary
    label_lookup = {}
    for _, row in label_df.iterrows():
        sample_name = str(row['sample_name'])
        roi_name = str(row['roi_name'])

        label_info = {
            'gfp_status': row.get('gfp_status', 'unknown'),
            'sample_name': sample_name,
            'roi_name': roi_name
        }

        # Store with simple key
        key = f"{sample_name}-{roi_name}"
        label_lookup[key] = label_info

    # Regex pattern to extract sample_name and roi_name from filename
    # Pattern: cell-{sample_name}-ch{channel}-{roi_name}.{extension}
    # Example: cell-r05c03-f08-ch02-ID0008.tiff
    filename_pattern = re.compile(r'cell-([^-]+-[^-]+)-ch\d+-([^.]+)\.')

    # Match images to labels
    matched_data = {}
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}

    for file_path in image_dir.rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            filename = file_path.name

            # Extract sample_name and roi_name using regex
            match = filename_pattern.match(filename)
            if match:
                sample_name = match.group(1)  # e.g., 'r05c03-f21'
                roi_name = match.group(2)     # e.g., 'ID0017'

                # Direct lookup in label_lookup dictionary
                lookup_key = f"{sample_name}-{roi_name}"
                if lookup_key in label_lookup:
                    matched_data[str(file_path)] = label_lookup[lookup_key]
                    logger.debug(f"Matched {filename}: {sample_name} + {roi_name} -> {label_lookup[lookup_key]['gfp_status']}")
                else:
                    logger.warning(f"No label data found for extracted key: {lookup_key} from {filename}")
                    matched_data[str(file_path)] = {
                        'gfp_status': 'unknown',
                        'sample_name': sample_name,
                        'roi_name': roi_name
                    }
            else:
                logger.warning(f"Could not extract sample_name and roi_name from filename: {filename}")
                matched_data[str(file_path)] = {
                    'gfp_status': 'unknown',
                    'sample_name': 'unknown',
                    'roi_name': 'unknown'
                }

    # Log matching statistics
    matched_count = sum(1 for data in matched_data.values() if data['gfp_status'] != 'unknown')
    unmatched_count = len(matched_data) - matched_count
    logger.info(f"Matched {len(matched_data)} images total")
    logger.info(f"Successfully matched: {matched_count}, Unmatched: {unmatched_count}")

    return matched_data


def process_image_labeling(input_dir: Path, output_dir: Path, label_csv_path: Path) -> Dict:
    """
    Process images by adding label information to filenames and saving to output directory.

    Args:
        input_dir: Input directory with images
        output_dir: Output directory for labeled images
        label_csv_path: Path to parquet/CSV file with sample_name, roi_name, and gfp_status

    Returns:
        Dictionary with processing statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    label_csv_path = Path(label_csv_path)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing images from {input_dir} to {output_dir}")

    # Load label data
    label_df = load_label_data(label_csv_path)

    # Match images to labels
    image_label_mapping = match_images_to_labels(input_dir, label_df)

    # Process images
    processed_count = 0
    error_count = 0
    skipped_count = 0
    label_distribution = {}

    for original_path, label_info in image_label_mapping.items():
        try:
            original_path = Path(original_path)
            gfp_status = label_info['gfp_status']

            if gfp_status == 'unknown':
                logger.warning(f"Skipping image with unknown label: {original_path}")
                skipped_count += 1
                continue

            # Create new filename with label
            new_filename = add_label_to_filename(original_path.name, gfp_status)

            # Maintain directory structure in output
            relative_path = original_path.relative_to(input_dir)
            output_file_path = output_dir / relative_path.parent / new_filename
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Move file to output directory with new name (copy + delete original)
            shutil.copy2(original_path, output_file_path)
            original_path.unlink()  # Delete original file
            logger.debug(f"Moved: {original_path} -> {output_file_path}")

            # Track statistics
            processed_count += 1
            label_distribution[gfp_status] = label_distribution.get(gfp_status, 0) + 1

            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count} images...")

        except Exception as e:
            logger.error(f"Error processing {original_path}: {e}")
            error_count += 1

    # Return simple statistics
    stats = {
        "processed_count": processed_count,
        "error_count": error_count,
        "skipped_count": skipped_count,
        "label_distribution": label_distribution
    }

    logger.info(f"Labeling complete. Processed: {processed_count}, Errors: {error_count}, Skipped: {skipped_count}")
    logger.info(f"Label distribution: {label_distribution}")

    return stats


if __name__ == "__main__":
    input_directory = Path("data/03_primary")
    output_directory = Path("data/03_primary")  # Same as input for in-place renaming
    label_parquet = Path("data/03_primary/combined_cell_data_clean.parquet")

    # Process images with labeling
    stats = process_image_labeling(
        input_dir=input_directory,
        output_dir=output_directory,
        label_csv_path=label_parquet
    )

    print(f"Labeling completed: {stats}")