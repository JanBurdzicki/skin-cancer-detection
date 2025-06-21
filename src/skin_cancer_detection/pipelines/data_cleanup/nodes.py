"""
Nodes for data cleanup pipeline.
"""

import logging
from pathlib import Path

from skin_cancer_detection.utils.cleanup_invalid_data import cleanup_invalid_data

logger = logging.getLogger(__name__)


def cleanup_invalid_data_node(patterns_file_path: str, dry_run: bool = True) -> None:
    """
    Kedro node to clean up invalid data based on patterns.

    Args:
        patterns_file_path: Path to file containing invalid patterns
        dry_run: If True, only log what would be removed
    """
    logger.info("Starting data cleanup node...")

    patterns_file = Path(patterns_file_path)
    cleanup_invalid_data(patterns_file, dry_run=dry_run)

    logger.info("Data cleanup node completed!")


# For dry run (default)
def cleanup_invalid_data_dry_run_node(patterns_file_path: str) -> None:
    """
    Kedro node to perform dry run cleanup of invalid data.

    Args:
        patterns_file_path: Path to file containing invalid patterns
    """
    cleanup_invalid_data_node(patterns_file_path, dry_run=True)


# For actual execution
def cleanup_invalid_data_execute_node(patterns_file_path: str) -> None:
    """
    Kedro node to execute cleanup of invalid data.

    Args:
        patterns_file_path: Path to file containing invalid patterns
    """
    cleanup_invalid_data_node(patterns_file_path, dry_run=False)