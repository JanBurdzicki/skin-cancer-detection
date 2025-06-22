#!/usr/bin/env python3
"""
Script to add all untracked data files and folders to DVC.

Usage:
    python3 scripts/dvc_add.py [directory]

    directory: Optional path to the directory to scan (default: data/01_raw)

This script:
- Scans the specified directory for files and directories.
- Skips items that are already tracked by DVC (.dvc file exists).
- Skips hidden files and directories (starting with .).
- Runs `dvc add <item>` for each untracked file or directory.
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
        description="Add untracked data files and folders to DVC",
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
        help=f"Directory to scan for untracked files and folders (default: {DEFAULT_DATA_DIR})"
    )
    return parser.parse_args()


def is_dvc_tracked(item: Path) -> bool:
    """
    Check if a DVC tracking file (.dvc) exists for the given item.

    Args:
        item (Path): Path to a data file or folder.

    Returns:
        bool: True if a .dvc file exists, False otherwise.
    """
    dvc_file = item.with_suffix(".dvc")
    return dvc_file.exists()


def dvc_add(item: Path):
    """
    Run `dvc add` on the specified file or folder.

    Args:
        item (Path): File or folder to add to DVC.
    """
    item_type = "directory" if item.is_dir() else "file"
    logging.info(f"Adding {item_type} to DVC: {item}")
    try:
        subprocess.run(["dvc", "add", str(item)], check=True)
        logging.info(f"Successfully added {item} to DVC")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to add {item} to DVC. Error: {e}")


def main():
    """
    Main function to scan and add untracked files and folders to DVC.
    """
    setup_logging()

    args = parse_arguments()
    data_dir = Path(args.directory)

    if not data_dir.exists():
        logging.error(f"Directory {data_dir} does not exist.")
        return

    logging.info(f"Scanning directory: {data_dir}")

    # Process both files and directories
    for item in sorted(data_dir.iterdir()):
        # Skip hidden files and directories (starting with .)
        if item.name.startswith('.'):
            logging.debug(f"Skipping hidden item: {item}")
            continue

        if is_dvc_tracked(item):
            item_type = "directory" if item.is_dir() else "file"
            logging.info(f"Already tracked by DVC, skipping {item_type}: {item}")
        else:
            dvc_add(item)


if __name__ == "__main__":
    main()
