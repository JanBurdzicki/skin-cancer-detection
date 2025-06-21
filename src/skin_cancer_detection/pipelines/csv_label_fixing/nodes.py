"""
Nodes for the CSV label fixing pipeline.
"""

import logging
from typing import Dict, Any

from skin_cancer_detection.utils.fix_csv_labels import fix_all_csv_labels

logger = logging.getLogger(__name__)


def fix_csv_labels_node() -> Dict[str, Any]:
    """
    Node to fix CSV labels across all samples.

    Returns:
        Dictionary with processing results
    """
    logger.info("Starting CSV label fixing process...")

    try:
        fix_all_csv_labels()

        return {
            "status": "success",
            "message": "CSV labels fixed successfully"
        }

    except Exception as e:
        logger.error(f"CSV label fixing failed: {e}")
        return {
            "status": "error",
            "message": f"CSV label fixing failed: {e}"
        }