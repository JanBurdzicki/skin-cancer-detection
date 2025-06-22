"""
Nodes for image labeling pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from skin_cancer_detection.utils.image_labeling import (
    process_image_labeling,
    load_label_data
)

logger = logging.getLogger(__name__)


def image_labeling_node(
    labeling_input_dir: str,
    labeling_output_dir: str,
    label_csv_path: str
) -> Dict[str, Any]:
    """
    Kedro node for image labeling process.

    Args:
        labeling_input_dir: Path to directory containing images
        labeling_output_dir: Path to output directory for labeled images
        label_csv_path: Path to parquet/CSV file with sample_name, roi_name, gfp_status

    Returns:
        Dictionary with processing statistics and metadata
    """
    # Convert paths to Path objects
    input_path = Path(labeling_input_dir)
    output_path = Path(labeling_output_dir)
    label_path = Path(label_csv_path)

    # Log configuration
    logger.info(f"Starting image labeling process")
    logger.info(f"Input directory: {input_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Label data: {label_path}")

    # Run the labeling process
    stats = process_image_labeling(
        input_dir=input_path,
        output_dir=output_path,
        label_csv_path=label_path
    )

    logger.info(f"Image labeling completed successfully")
    logger.info(f"Final statistics: {stats}")

    return stats


def validate_label_data_node(label_csv_path: str) -> Dict[str, Any]:
    """
    Validate the label data before processing.

    Args:
        label_csv_path: Path to parquet/CSV with sample_name, roi_name, gfp_status

    Returns:
        Dictionary with validation results
    """
    csv_path = Path(label_csv_path)

    logger.info(f"Validating label data from {csv_path}")

    # Load and validate label data
    label_df = load_label_data(csv_path)

    # Get distribution stats
    label_distribution = label_df['gfp_status'].value_counts().to_dict()

    # Check for class imbalance
    total_samples = sum(label_distribution.values())
    class_balance = {}
    for label, count in label_distribution.items():
        class_balance[label] = {
            'count': count,
            'percentage': (count / total_samples) * 100 if total_samples > 0 else 0
        }

    validation_results = {
        "total_samples": len(label_df),
        "label_distribution": label_distribution,
        "class_balance": class_balance,
        "unique_labels": list(label_distribution.keys()),
        "is_balanced": all(0.3 <= info['percentage'] <= 0.7 for info in class_balance.values()) if len(class_balance) == 2 else None
    }

    logger.info(f"Label validation results: {validation_results}")

    return validation_results