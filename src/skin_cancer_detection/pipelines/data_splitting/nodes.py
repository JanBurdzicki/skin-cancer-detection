"""
Nodes for data splitting pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from skin_cancer_detection.utils.data_splitting import (
    process_data_splitting,
    load_feature_data,
    stratified_split_data
)

logger = logging.getLogger(__name__)


def data_splitting_node(
    feature_data_path: str,
    image_data_dir: str,
    output_dir: str,
    data_splitting_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Kedro node for data splitting process.

    Args:
        feature_data_path: Path to feature data parquet file
        image_data_dir: Directory containing image files
        output_dir: Output directory for data splits
        data_splitting_config: Configuration dictionary for data splitting

    Returns:
        Dictionary with processing statistics and metadata
    """
    # Extract configuration parameters
    config = data_splitting_config or {}

    # Split configuration
    test_size = config.get('test_size', 0.2)
    val_size = config.get('val_size', 0.2)
    random_state = config.get('random_state', 42)

    # Column configuration
    label_col = config.get('label_column', 'gfp_status')
    sample_col = config.get('sample_column', 'sample_name')
    roi_col = config.get('roi_column', 'roi_name')

    # Image filtering configuration
    channel_filter = config.get('channel_filter', 'ch03')

    # Log configuration
    logger.info(f"Starting data splitting process")
    logger.info(f"Feature data: {feature_data_path}")
    logger.info(f"Image data: {image_data_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Split configuration: test={test_size}, val={val_size}, random_state={random_state}")

    # Run the data splitting process
    stats = process_data_splitting(
        input_file=feature_data_path,
        output_dir=output_dir,
        image_data_dir=image_data_dir,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        label_col=label_col,
        sample_col=sample_col,
        roi_col=roi_col,
        channel_filter=channel_filter
    )

    logger.info(f"Data splitting completed successfully")
    logger.info(f"Final statistics: {stats}")

    return stats


def validate_input_data_node(
    feature_data_path: str,
    image_data_dir: str
) -> Dict[str, Any]:
    """
    Validate the input data before processing.

    Args:
        feature_data_path: Path to feature data parquet file
        image_data_dir: Directory containing image files

    Returns:
        Dictionary with validation results
    """
    feature_path = Path(feature_data_path)
    image_path = Path(image_data_dir)

    logger.info(f"Validating input data")
    logger.info(f"Feature data: {feature_path}")
    logger.info(f"Image data: {image_path}")

    validation_results = {
        "feature_data": {
            "exists": feature_path.exists(),
            "size_mb": feature_path.stat().st_size / (1024 * 1024) if feature_path.exists() else 0
        },
        "image_data": {
            "exists": image_path.exists(),
            "is_directory": image_path.is_dir() if image_path.exists() else False
        }
    }

    if feature_path.exists():
        try:
            # Load and validate feature data
            df = load_feature_data(feature_path)

            # Check for required columns
            required_cols = ['sample_name', 'roi_name', 'gfp_status']
            missing_cols = [col for col in required_cols if col not in df.columns]

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

            validation_results["feature_data"].update({
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "required_columns_present": len(missing_cols) == 0,
                "missing_columns": missing_cols,
                "label_distribution": label_distribution,
                "class_balance": class_balance,
                "is_imbalanced": any(info['percentage'] < 10 or info['percentage'] > 90 for info in class_balance.values()) if len(class_balance) >= 2 else None
            })

        except Exception as e:
            logger.error(f"Error validating feature data: {e}")
            validation_results["feature_data"]["validation_error"] = str(e)

    if image_path.exists() and image_path.is_dir():
        try:
            # Count image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
            image_files = [f for f in image_path.rglob('*') if f.suffix.lower() in image_extensions]

            validation_results["image_data"].update({
                "total_images": len(image_files),
                "subdirectories": len([d for d in image_path.iterdir() if d.is_dir()]),
                "image_extensions": list(set(f.suffix.lower() for f in image_files))
            })

        except Exception as e:
            logger.error(f"Error validating image data: {e}")
            validation_results["image_data"]["validation_error"] = str(e)

    logger.info(f"Input validation results:")
    logger.info(f"  - Feature data: {validation_results['feature_data']}")
    logger.info(f"  - Image data: {validation_results['image_data']}")

    return validation_results


def validate_output_data_node(output_dir: str) -> Dict[str, Any]:
    """
    Validate the output data after processing.

    Args:
        output_dir: Output directory with data splits

    Returns:
        Dictionary with validation results
    """
    output_path = Path(output_dir)

    logger.info(f"Validating output data at {output_path}")

    if not output_path.exists():
        logger.warning(f"Output directory does not exist: {output_path}")
        return {
            "directory_exists": False,
            "validation_status": "failed",
            "message": "Output directory not found"
        }

    validation_results = {
        "directory_exists": True,
        "tabular_data": {},
        "image_data": {},
        "validation_status": "unknown"
    }

    # Check tabular data (saved in tabular/ subdirectory)
    tabular_dir = output_path / 'tabular'
    splits = ['train', 'val', 'test']
    tabular_files = {}

    if tabular_dir.exists():
        for split in splits:
            parquet_file = tabular_dir / f'{split}.parquet'

            if parquet_file.exists():
                tabular_files[split] = str(parquet_file)
                # Load and check
                try:
                    df = pd.read_parquet(parquet_file)
                    tabular_files[f'{split}_shape'] = df.shape
                    tabular_files[f'{split}_labels'] = df['gfp_status'].value_counts().to_dict() if 'gfp_status' in df.columns else {}
                except Exception as e:
                    tabular_files[f'{split}_error'] = str(e)

    validation_results["tabular_data"] = tabular_files

    # Check image data (organized in images/ subdirectory by split and class)
    images_dir = output_path / 'images'
    image_info = {}

    if images_dir.exists():
        for split in splits:
            split_dir = images_dir / split
            if split_dir.exists():
                # Count images by class
                class_counts = {}
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
                        image_count = len([f for f in class_dir.iterdir() if f.suffix.lower() in image_extensions])
                        class_counts[class_dir.name] = image_count

                image_info[split] = {
                    'directory': str(split_dir),
                    'class_counts': class_counts,
                    'total_images': sum(class_counts.values())
                }

    validation_results["image_data"] = image_info

    # Determine validation status
    has_tabular = bool(validation_results["tabular_data"])
    has_images = bool(validation_results["image_data"])

    if has_tabular and has_images:
        validation_results["validation_status"] = "passed"
    elif has_tabular or has_images:
        validation_results["validation_status"] = "partial"
    else:
        validation_results["validation_status"] = "failed"

    logger.info(f"Output validation results:")
    logger.info(f"  - Status: {validation_results['validation_status']}")
    logger.info(f"  - Has tabular data: {has_tabular}")
    logger.info(f"  - Has image data: {has_images}")

    return validation_results