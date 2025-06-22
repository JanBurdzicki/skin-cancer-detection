"""
Nodes for data filtering pipeline.
"""

import logging
from typing import Dict, Any

from skin_cancer_detection.utils.data_filtering import filter_data

logger = logging.getLogger(__name__)


def filter_data_node() -> Dict[str, Any]:
    """
    Kedro node to filter data based on missing values and sample/ROI combinations.

    Returns:
        Dictionary with filtering results
    """
    logger.info("Starting data filtering node...")

    results = filter_data()

    logger.info("Data filtering node completed!")
    return results