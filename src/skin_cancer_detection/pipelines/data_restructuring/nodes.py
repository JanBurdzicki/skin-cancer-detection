"""
Data restructuring pipeline nodes for skin cancer detection project.
"""

import logging
from pathlib import Path
from typing import Dict, Any

from skin_cancer_detection.utils.restructure import restructure_data

logger = logging.getLogger(__name__)


def restructure_raw_data() -> Dict[str, Any]:
    """
    Node to restructure raw data from scattered files into organized structure.

    Returns:
        Dictionary with restructuring status and metadata
    """
    logger.info("Starting data restructuring...")

    try:
        restructure_data()

        # Count processed samples
        intermediate_dir = Path("data/02_intermediate")
        samples = [d for d in intermediate_dir.glob("r*") if d.is_dir()]

        return {
            "status": "success",
            "samples_processed": len(samples),
            "sample_ids": [s.name for s in samples],
            "message": "Data restructuring completed successfully"
        }

    except Exception as e:
        logger.error(f"Data restructuring failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }


def validate_restructured_data() -> Dict[str, Any]:
    """
    Node to validate the restructured data and generate summary statistics.

    Returns:
        Dictionary with validation results and data summary
    """
    logger.info("Validating restructured data...")

    try:
        intermediate_dir = Path("data/02_intermediate")

        validation_results = {
            "status": "success",
            "intermediate_exists": intermediate_dir.exists(),
            "samples": [],
            "file_summary": {
                "total_csv_files": 0,
                "total_parquet_files": 0,
                "total_tiff_files": 0
            }
        }

        # Check samples in intermediate directory
        for sample_dir in intermediate_dir.glob("r*"):
            if sample_dir.is_dir():
                sample_info = {
                    "sample_id": sample_dir.name,
                    "has_images": (sample_dir / "images").exists(),
                    "has_patches": (sample_dir / "patches").exists(),
                    "has_stats": (sample_dir / "stats").exists(),
                }

                # Count files in each directory
                if sample_info["has_images"]:
                    sample_info["image_count"] = len(list((sample_dir / "images").glob("*.tiff")))
                    validation_results["file_summary"]["total_tiff_files"] += sample_info["image_count"]

                if sample_info["has_patches"]:
                    patch_count = 0
                    for ch_dir in (sample_dir / "patches").glob("*"):
                        if ch_dir.is_dir():
                            patch_count += len(list(ch_dir.glob("*.tiff")))
                    sample_info["patch_count"] = patch_count
                    validation_results["file_summary"]["total_tiff_files"] += patch_count

                if sample_info["has_stats"]:
                    csv_count = len(list((sample_dir / "stats").rglob("*.csv")))
                    parquet_count = len(list((sample_dir / "stats").rglob("*.parquet")))
                    sample_info["csv_count"] = csv_count
                    sample_info["parquet_count"] = parquet_count
                    validation_results["file_summary"]["total_csv_files"] += csv_count
                    validation_results["file_summary"]["total_parquet_files"] += parquet_count

                validation_results["samples"].append(sample_info)

        return validation_results

    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }