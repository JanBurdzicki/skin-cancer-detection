"""
CSV to Parquet conversion pipeline nodes.
"""

import logging
from pathlib import Path
from typing import Dict, Any

from skin_cancer_detection.utils.convert_csv_to_parquet import convert_csv_to_parquet

logger = logging.getLogger(__name__)


def convert_csvs_to_parquet() -> Dict[str, Any]:
    """
    Node to convert all CSV files to Parquet format for better performance.
    Converts CSV files within 02_intermediate directory, maintaining the same structure.

    Returns:
        Dictionary with conversion status and statistics
    """
    logger.info("Converting CSV files to Parquet format...")

    try:
        base_dir = Path("data/02_intermediate")

        # Count CSV files before conversion
        csv_files = list(base_dir.rglob("*.csv"))
        csv_count = len(csv_files)

        # Convert CSV files to Parquet in the same directory structure
        converted_count = convert_csv_to_parquet(base_dir, base_dir, preserve_structure=True, in_place=True)

        # Count Parquet files after conversion
        parquet_files = list(base_dir.rglob("*.parquet"))
        parquet_count = len(parquet_files)

        return {
            "status": "success",
            "csv_files_found": csv_count,
            "files_converted": converted_count,
            "parquet_files_total": parquet_count,
            "base_dir": str(base_dir),
            "message": f"Converted {converted_count} CSV files to Parquet format in {base_dir}"
        }

    except Exception as e:
        logger.error(f"CSV to Parquet conversion failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }