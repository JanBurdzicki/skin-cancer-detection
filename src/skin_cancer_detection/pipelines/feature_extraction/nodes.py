"""
Nodes for feature extraction pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from skin_cancer_detection.utils.feature_extraction import (
    process_feature_extraction,
    load_cell_data,
    filter_feature_columns
)

logger = logging.getLogger(__name__)


def feature_extraction_node(
    input_data_path: str,
    output_data_path: str,
    feature_extraction_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Kedro node for feature extraction process.

    Args:
        input_data_path: Path to input parquet file with cell data
        output_data_path: Path to output parquet file for feature data
        feature_extraction_config: Configuration dictionary for feature extraction

    Returns:
        Dictionary with processing statistics and metadata
    """
    # Convert paths to Path objects
    input_path = Path(input_data_path)
    output_path = Path(output_data_path)

    # Extract configuration parameters
    config = feature_extraction_config or {}
    metadata_cols = config.get('metadata_columns', ['sample_name', 'roi_name'])
    feature_prefix = config.get('feature_prefix', 'mt_')
    label_col = config.get('label_column', 'gfp_status')

    # Log configuration
    logger.info(f"Starting feature extraction process")
    logger.info(f"Input file: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Configuration: metadata_cols={metadata_cols}, feature_prefix='{feature_prefix}', label_col='{label_col}'")

    # Run the feature extraction process
    stats = process_feature_extraction(
        input_path=input_path,
        output_path=output_path,
        metadata_cols=metadata_cols,
        feature_prefix=feature_prefix,
        label_col=label_col
    )

    logger.info(f"Feature extraction completed successfully")
    logger.info(f"Final statistics: {stats}")

    return stats


def validate_input_data_node(input_data_path: str) -> Dict[str, Any]:
    """
    Validate the input data before processing.

    Args:
        input_data_path: Path to input parquet file with cell data

    Returns:
        Dictionary with validation results
    """
    input_path = Path(input_data_path)

    logger.info(f"Validating input data from {input_path}")

    # Load and validate input data
    df = load_cell_data(input_path)

    # Check for required columns
    required_cols = ['sample_name', 'roi_name', 'gfp_status']
    missing_cols = [col for col in required_cols if col not in df.columns]

    # Find mt_ columns
    mt_cols = [col for col in df.columns if col.startswith('mt_')]

    # Get label distribution
    label_distribution = df['gfp_status'].value_counts().to_dict() if 'gfp_status' in df.columns else {}

    # Check for class imbalance
    total_samples = sum(label_distribution.values()) if label_distribution else 0
    class_balance = {}
    for label, count in label_distribution.items():
        class_balance[label] = {
            'count': count,
            'percentage': (count / total_samples) * 100 if total_samples > 0 else 0
        }

    validation_results = {
        "file_exists": input_path.exists(),
        "file_size_mb": input_path.stat().st_size / (1024 * 1024) if input_path.exists() else 0,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "required_columns_present": len(missing_cols) == 0,
        "missing_columns": missing_cols,
        "mt_feature_columns_count": len(mt_cols),
        "mt_feature_columns": mt_cols,
        "label_distribution": label_distribution,
        "class_balance": class_balance,
        "unique_labels": list(label_distribution.keys()),
        "has_data": len(df) > 0,
        "is_balanced": all(0.2 <= info['percentage'] <= 0.8 for info in class_balance.values()) if len(class_balance) >= 2 else None
    }

    logger.info(f"Input data validation results:")
    logger.info(f"  - File exists: {validation_results['file_exists']}")
    logger.info(f"  - Shape: ({validation_results['total_rows']}, {validation_results['total_columns']})")
    logger.info(f"  - Required columns present: {validation_results['required_columns_present']}")
    logger.info(f"  - MT feature columns: {validation_results['mt_feature_columns_count']}")
    logger.info(f"  - Label distribution: {validation_results['label_distribution']}")

    return validation_results


def validate_output_data_node(output_data_path: str) -> Dict[str, Any]:
    """
    Validate the output data after processing.

    Args:
        output_data_path: Path to output parquet file with feature data

    Returns:
        Dictionary with validation results
    """
    output_path = Path(output_data_path)

    logger.info(f"Validating output data at {output_path}")

    if not output_path.exists():
        logger.warning(f"Output file does not exist: {output_path}")
        return {
            "file_exists": False,
            "validation_status": "failed",
            "message": "Output file not found"
        }

    # Load and validate output data
    df = pd.read_parquet(output_path)

    # Expected columns
    expected_metadata = ['sample_name', 'roi_name']
    expected_label = ['gfp_status']
    mt_cols = [col for col in df.columns if col.startswith('mt_')]

    # Validation checks
    has_metadata = all(col in df.columns for col in expected_metadata)
    has_label = all(col in df.columns for col in expected_label)
    has_features = len(mt_cols) > 0

    validation_results = {
        "file_exists": True,
        "file_size_mb": output_path.stat().st_size / (1024 * 1024),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "has_metadata_columns": has_metadata,
        "metadata_columns": expected_metadata,
        "has_label_column": has_label,
        "label_column": expected_label,
        "has_feature_columns": has_features,
        "feature_columns_count": len(mt_cols),
        "feature_columns": mt_cols,
        "validation_status": "passed" if (has_metadata and has_label and has_features) else "failed",
        "label_distribution": df['gfp_status'].value_counts().to_dict() if 'gfp_status' in df.columns else {}
    }

    logger.info(f"Output data validation results:")
    logger.info(f"  - Shape: ({validation_results['total_rows']}, {validation_results['total_columns']})")
    logger.info(f"  - Validation status: {validation_results['validation_status']}")
    logger.info(f"  - Feature columns: {validation_results['feature_columns_count']}")

    return validation_results