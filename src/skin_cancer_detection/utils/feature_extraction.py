"""
Utility script for extracting features from cell data.

This module provides functions to:
1. Load cell data from parquet files
2. Filter specific columns (metadata, features starting with 'mt_', and labels)
3. Save processed feature data for machine learning pipelines
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

logger = logging.getLogger(__name__)


def load_cell_data(input_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load cell data from parquet file.

    Args:
        input_path: Path to parquet file with cell data

    Returns:
        DataFrame with cell data
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise ValueError(f"File does not exist: {input_path}")

    if input_path.suffix.lower() != '.parquet':
        raise ValueError(f"Expected parquet file, got: {input_path.suffix}")

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded cell data with {len(df)} records and {len(df.columns)} columns from {input_path}")

    return df


def filter_feature_columns(
    df: pd.DataFrame,
    metadata_cols: List[str] = None,
    feature_prefix: str = 'mt_',
    label_col: str = 'gfp_status'
) -> pd.DataFrame:
    """
    Filter and select specific columns for feature extraction.

    Args:
        df: Input DataFrame with cell data
        metadata_cols: List of metadata column names (default: ['sample_name', 'roi_name'])
        feature_prefix: Prefix for feature columns (default: 'mt_')
        label_col: Label column name (default: 'gfp_status')

    Returns:
        DataFrame with filtered columns
    """
    # Set defaults if not provided
    if metadata_cols is None:
        metadata_cols = ['sample_name', 'roi_name']

    # Check for required columns
    missing_cols = []
    for col in metadata_cols + [label_col]:
        if col not in df.columns:
            missing_cols.append(col)

    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")

        # Find all columns starting with the specified prefix
    feature_cols = [col for col in df.columns if col.startswith(feature_prefix)]

    if not feature_cols:
        raise ValueError(f"No columns starting with '{feature_prefix}' found in the data")

    # Combine all columns to keep
    columns_to_keep = metadata_cols + feature_cols + [label_col]

    # Filter the DataFrame
    filtered_df = df[columns_to_keep].copy()

    logger.info(f"Filtered data to {len(columns_to_keep)} columns:")
    logger.info(f"  - Metadata columns: {len(metadata_cols)} ({metadata_cols})")
    logger.info(f"  - Feature columns: {len(feature_cols)} (starting with '{feature_prefix}')")
    logger.info(f"  - Label column: 1 ({label_col})")

    # Log label distribution
    if label_col in filtered_df.columns:
        label_counts = filtered_df[label_col].value_counts()
        logger.info(f"Label distribution: {label_counts.to_dict()}")

    return filtered_df


def save_feature_data(df: pd.DataFrame, output_path: Union[str, Path]) -> None:
    """
    Save processed feature data to parquet file.

    Args:
        df: DataFrame with processed feature data
        output_path: Path where to save the processed data
    """
    output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as parquet
    df.to_parquet(output_path, index=False)

    logger.info(f"Saved processed feature data to {output_path}")
    logger.info(f"Output file size: {output_path.stat().st_size / 1024:.1f} KB")


def process_feature_extraction(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    metadata_cols: List[str] = None,
    feature_prefix: str = 'mt_',
    label_col: str = 'gfp_status'
) -> Dict:
    """
    Complete feature extraction pipeline: load data, filter columns, and save results.

    Args:
        input_path: Path to input parquet file with cell data
        output_path: Path where to save processed feature data
        metadata_cols: List of metadata column names (default: ['sample_name', 'roi_name'])
        feature_prefix: Prefix for feature columns (default: 'mt_')
        label_col: Label column name (default: 'gfp_status')

    Returns:
        Dictionary with processing statistics
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    logger.info(f"Starting feature extraction pipeline")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")

    # Load data
    df = load_cell_data(input_path)
    original_shape = df.shape

    # Filter columns
    filtered_df = filter_feature_columns(df, metadata_cols, feature_prefix, label_col)
    filtered_shape = filtered_df.shape

    # Save processed data
    save_feature_data(filtered_df, output_path)

    # Create statistics
    stats = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "original_shape": original_shape,
        "filtered_shape": filtered_shape,
        "rows_processed": filtered_shape[0],
        "columns_kept": filtered_shape[1],
        "metadata_columns": len(metadata_cols or ['sample_name', 'roi_name']),
        "feature_columns": len([col for col in filtered_df.columns if col.startswith(feature_prefix)]),
        "label_column": 1,  # Always one label column
        "label_distribution": filtered_df[label_col].value_counts().to_dict() if label_col in filtered_df.columns else {}
    }

    logger.info(f"Feature extraction completed successfully")
    logger.info(f"Processed {stats['rows_processed']} rows with {stats['columns_kept']} columns")
    logger.info(f"Features: {stats['feature_columns']} {feature_prefix}* columns, Metadata: {stats['metadata_columns']}, Label: {stats['label_column']}")

    return stats


if __name__ == "__main__":
    # Default paths
    input_file = Path("data/03_primary/combined_cell_data_clean.parquet")
    output_file = Path("data/04_feature/feature_data.parquet")

    # Run feature extraction
    stats = process_feature_extraction(
        input_path=input_file,
        output_path=output_file
    )

    print(f"Feature extraction completed: {stats}")