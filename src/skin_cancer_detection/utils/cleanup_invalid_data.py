"""
Utility for cleaning up invalid images and patches.

This script removes images and patches that contain invalid patterns
(e.g., patterns from failed image analysis).
"""

import argparse
import logging
from pathlib import Path
from typing import List, Set

# Setup logging
logger = logging.getLogger(__name__)

# Config
INTERMEDIATE_ROOT = Path("data/02_intermediate")
RAW_ROOT = Path("data/01_raw")


def load_invalid_patterns(patterns_file: Path) -> Set[str]:
    """
    Load invalid patterns from file.

    Args:
        patterns_file: Path to file containing invalid patterns

    Returns:
        Set of invalid patterns
    """
    if not patterns_file.exists():
        logger.warning(f"Patterns file does not exist: {patterns_file}")
        return set()

    patterns = set()
    try:
        with open(patterns_file, 'r') as f:
            for line in f:
                pattern = line.strip()
                if pattern:
                    patterns.add(pattern)

        logger.info(f"Loaded {len(patterns)} invalid patterns from {patterns_file}")
        return patterns

    except Exception as e:
        logger.error(f"Failed to load patterns from {patterns_file}: {e}")
        return set()


def find_files_with_patterns(root_dir: Path, patterns: Set[str], extensions: List[str]) -> List[Path]:
    """
    Find all files containing any of the patterns.

    Args:
        root_dir: Root directory to search
        patterns: Set of patterns to match
        extensions: List of file extensions to consider

    Returns:
        List of file paths to remove
    """
    files_to_remove = []

    if not root_dir.exists():
        logger.warning(f"Directory does not exist: {root_dir}")
        return files_to_remove

    for ext in extensions:
        for file_path in root_dir.rglob(f"*{ext}"):
            filename = file_path.name

            # Check if filename contains any invalid pattern
            for pattern in patterns:
                if pattern in filename:
                    files_to_remove.append(file_path)
                    logger.debug(f"Found file to remove: {file_path} (pattern: {pattern})")
                    break

    return files_to_remove


def remove_files(files_to_remove: List[Path], dry_run: bool = True) -> int:
    """
    Remove the specified files.

    Args:
        files_to_remove: List of file paths to remove
        dry_run: If True, only log what would be removed

    Returns:
        Number of files actually removed
    """
    if not files_to_remove:
        logger.info("No files to remove")
        return 0

    removed_count = 0

    for file_path in files_to_remove:
        try:
            if dry_run:
                logger.info(f"Would remove: {file_path}")
            else:
                file_path.unlink()
                logger.info(f"Removed: {file_path}")
                removed_count += 1

        except Exception as e:
            logger.error(f"Failed to remove {file_path}: {e}")

    if dry_run:
        logger.info(f"Dry run completed. Would remove {len(files_to_remove)} files")
    else:
        logger.info(f"Removed {removed_count} files")

    return removed_count


def cleanup_invalid_data(patterns_file: Path, dry_run: bool = True) -> None:
    """
    Clean up invalid images and patches based on patterns.

    Args:
        patterns_file: Path to file containing invalid patterns
        dry_run: If True, only log what would be removed
    """
    logger.info(f"Starting cleanup of invalid data (dry_run={dry_run})")

    # Load invalid patterns
    patterns = load_invalid_patterns(patterns_file)
    if not patterns:
        logger.info("No patterns to process")
        return

    logger.info(f"Processing {len(patterns)} invalid patterns: {sorted(patterns)}")

    total_removed = 0

    # Clean up intermediate data
    logger.info("Cleaning up intermediate data...")

    # Remove restructured images
    tiff_files = find_files_with_patterns(INTERMEDIATE_ROOT, patterns, [".tiff", ".tif"])
    logger.info(f"Found {len(tiff_files)} TIFF files to remove in intermediate data")
    total_removed += remove_files(tiff_files, dry_run)

    # Remove patches
    patch_files = find_files_with_patterns(INTERMEDIATE_ROOT, patterns, [".png", ".jpg", ".jpeg"])
    logger.info(f"Found {len(patch_files)} patch files to remove in intermediate data")
    total_removed += remove_files(patch_files, dry_run)

    # Remove CSV files
    csv_files = find_files_with_patterns(INTERMEDIATE_ROOT, patterns, [".csv"])
    logger.info(f"Found {len(csv_files)} CSV files to remove in intermediate data")
    total_removed += remove_files(csv_files, dry_run)

    # Clean up raw data if it exists
    if RAW_ROOT.exists():
        logger.info("Cleaning up raw data...")

        raw_files = find_files_with_patterns(RAW_ROOT, patterns, [".tiff", ".tif", ".png", ".jpg", ".jpeg"])
        logger.info(f"Found {len(raw_files)} raw files to remove")
        total_removed += remove_files(raw_files, dry_run)

    # Remove empty directories
    if not dry_run:
        logger.info("Removing empty directories...")
        remove_empty_directories(INTERMEDIATE_ROOT)
        if RAW_ROOT.exists():
            remove_empty_directories(RAW_ROOT)

    if dry_run:
        logger.info(f"Cleanup dry run completed. Would remove {total_removed} files total")
    else:
        logger.info(f"Cleanup completed. Removed {total_removed} files total")


def remove_empty_directories(root_dir: Path) -> None:
    """
    Remove empty directories recursively.

    Args:
        root_dir: Root directory to start from
    """
    try:
        for dir_path in sorted(root_dir.rglob("*"), reverse=True):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                    logger.info(f"Removed empty directory: {dir_path}")
                except OSError:
                    pass  # Directory not empty or permission denied
    except Exception as e:
        logger.error(f"Error removing empty directories: {e}")


def main() -> None:
    """Main entry point for the cleanup script."""
    parser = argparse.ArgumentParser(
        description="Clean up invalid images and patches based on patterns"
    )
    parser.add_argument(
        "--patterns-file",
        type=Path,
        default=Path("data") / "invalid_patterns.txt",
        help="Path to file containing invalid patterns (default: data/invalid_patterns.txt)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be removed without actually removing files"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually remove the files (opposite of --dry-run)"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Determine dry_run mode
    dry_run = not args.execute if args.execute else True
    if args.dry_run:
        dry_run = True

    try:
        cleanup_invalid_data(args.patterns_file, dry_run=dry_run)

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise


if __name__ == "__main__":
    main()