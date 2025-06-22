#!/usr/bin/env python3
"""
Script to add all untracked data folders to DVC.

Usage:
    python3 scripts/dvc_add.py [directory]

    directory: Optional path to the directory to scan (default: data/01_raw)

This script:
- Recursively scans the specified directory for directories.
- Skips folders that are already tracked by DVC (.dvc file exists).
- Runs `dvc add <folder>` for each untracked directory.
- Uses subprocess with error handling.
"""

import os
import subprocess
import logging
import argparse
from pathlib import Path


# Constants
DEFAULT_DATA_DIR = Path("data/01_raw")
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def setup_logging():
    """Configure logging format and level."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Add untracked data folders to DVC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/dvc_add.py                    # Use default directory: data/01_raw
  python3 scripts/dvc_add.py data/02_intermediate  # Specify custom directory
        """
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=str(DEFAULT_DATA_DIR),
        help=f"Directory to scan for untracked folders (default: {DEFAULT_DATA_DIR})"
    )
    return parser.parse_args()


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
    Main function to scan and add untracked folders to DVC.
    """
    setup_logging()

    args = parse_arguments()
    data_dir = Path(args.directory)

    if not data_dir.exists():
        logging.error(f"Directory {data_dir} does not exist.")
        return

    logging.info(f"Scanning directory: {data_dir}")

    for item in sorted(data_dir.iterdir()):
        if item.is_dir():
            if is_dvc_tracked(item):
                logging.info(f"Already tracked by DVC, skipping: {item}")
            else:
                dvc_add(item)


if __name__ == "__main__":
    main()
