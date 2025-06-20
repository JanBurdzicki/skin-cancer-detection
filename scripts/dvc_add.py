#!/usr/bin/env python3
"""
Script to add all untracked data folders in `data/01_raw/` to DVC.

Usage:
    python3 scripts/dvc_add.py

This script:
- Recursively scans `data/01_raw/` for directories.
- Skips folders that are already tracked by DVC (.dvc file exists).
- Runs `dvc add <folder>` for each untracked directory.
- Uses subprocess with error handling.
"""

import os
import subprocess
import logging
from pathlib import Path


# Constants
RAW_DATA_DIR = Path("data/01_raw")
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def setup_logging():
    """Configure logging format and level."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def is_dvc_tracked(folder: Path) -> bool:
    """
    Check if a DVC tracking file (.dvc) exists for the given folder.

    Args:
        folder (Path): Path to a data subfolder.

    Returns:
        bool: True if a .dvc file exists, False otherwise.
    """
    dvc_file = folder.with_suffix(".dvc")
    return dvc_file.exists()


def dvc_add(folder: Path):
    """
    Run `dvc add` on the specified folder.

    Args:
        folder (Path): Folder to add to DVC.
    """
    logging.info(f"Adding to DVC: {folder}")
    try:
        subprocess.run(["dvc", "add", str(folder)], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to add {folder} to DVC. Error: {e}")


def main():
    """
    Main function to scan and add untracked folders in `data/01_raw/` to DVC.
    """
    setup_logging()

    if not RAW_DATA_DIR.exists():
        logging.error(f"Directory {RAW_DATA_DIR} does not exist.")
        return

    for item in sorted(RAW_DATA_DIR.iterdir()):
        if item.is_dir():
            if is_dvc_tracked(item):
                logging.info(f"Already tracked by DVC, skipping: {item}")
            else:
                dvc_add(item)


if __name__ == "__main__":
    main()
