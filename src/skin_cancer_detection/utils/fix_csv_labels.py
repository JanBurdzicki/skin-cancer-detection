"""
CSV label correction utilities for skin cancer detection project.

This module provides functions to fix the Label column in CSV files to match
the proper patch filenames instead of the original complex filenames.
"""

import csv
import logging
import re
from pathlib import Path
from typing import Dict, Optional

# Setup logging
logger = logging.getLogger(__name__)

# Config
INTERMEDIATE_ROOT = Path("data/02_intermediate")
CHANNEL_MAP = {
    "ch01": "dapi",
    "ch02": "gfp",
    "ch03": "mt",
    # ch04 is skipped as it's useless
}


def extract_cell_id_from_label(label: str) -> Optional[str]:
    """
    Extract cell ID from a complex label string.

    Args:
        label: Complex label like "dapi_0001 - ID 0048_r01c01f01p03-ch01t01.tiff:0001 - ID 0048"

    Returns:
        Formatted cell ID like "ID0048" or None if not found

    Example:
        >>> extract_cell_id_from_label("dapi_0001 - ID 0048_r01c01f01p03-ch01t01.tiff:0001 - ID 0048")
        'ID0048'
    """
    # Look for "ID ####" pattern in the label
    match = re.search(r'ID (\d+)', label)
    if match:
        return f"ID{match.group(1).zfill(4)}"

    return None


def extract_metadata_from_label(label: str) -> Dict[str, str]:
    """
    Extract metadata from a complex label string.

    Args:
        label: Complex label like "dapi_0001 - ID 0048_r01c01f01p03-ch01t01.tiff:0001 - ID 0048"

    Returns:
        Dictionary with extracted metadata

    Example:
        >>> extract_metadata_from_label("dapi_0001 - ID 0048_r01c01f01p03-ch01t01.tiff:0001 - ID 0048")
        {'row_col': 'r01c01', 'field': 'f01', 'channel': 'ch01', 'cell_id': 'ID0048'}
    """
    try:
        # Extract cell ID
        cell_id = extract_cell_id_from_label(label)

        # Extract the main part with coordinates
        # Look for pattern like "r##c##f##p##-ch##t##"
        coord_match = re.search(r'(r\d+c\d+)(f\d+)p\d+-(ch\d+)t\d+', label)
        if coord_match:
            row_col = coord_match.group(1)
            field = coord_match.group(2)
            channel = coord_match.group(3)

            return {
                'row_col': row_col,
                'field': field,
                'channel': channel,
                'cell_id': cell_id
            }
        else:
            logger.warning(f"Could not parse coordinates from label: {label}")
            return {'row_col': '', 'field': '', 'channel': '', 'cell_id': cell_id}

    except Exception as e:
        logger.error(f"Failed to extract metadata from label {label}: {e}")
        return {'row_col': '', 'field': '', 'channel': '', 'cell_id': None}


def generate_proper_filename(sample_id: str, field: str, channel: str, cell_id: str) -> str:
    """
    Generate the proper patch filename that should be referenced in the Label column.

    Args:
        sample_id: Sample identifier (e.g., 'r01c01')
        field: Field identifier (e.g., 'f01')
        channel: Channel identifier (e.g., 'ch01')
        cell_id: Cell identifier (e.g., 'ID0048')

    Returns:
        Proper filename like "cell-r01c01-f01-ch01-ID0048.tiff"
    """
    return f"cell-{sample_id}-{field}-{channel}-{cell_id}.tiff"


def is_label_already_correct(label: str, sample_id: str) -> bool:
    """
    Check if a label is already in the correct format.

    Args:
        label: Label to check
        sample_id: Expected sample ID (e.g., 'r05c03')

    Returns:
        True if label is already in correct format
    """
    # Check if label matches the pattern: cell-{sample_id}-{field}-{channel}-{cell_id}.tiff
    pattern = rf'cell-{re.escape(sample_id)}-f\d+-ch\d+-ID\d{{4}}\.tiff'
    return bool(re.match(pattern, label))


def collect_invalid_pattern_from_csv(csv_path: Path, sample_id: str) -> Optional[str]:
    """
    Check if CSV file is empty or doesn't exist, and return the invalid pattern.

    Args:
        csv_path: Path to the CSV file
        sample_id: Sample identifier (e.g., 'r01c01')

    Returns:
        Invalid pattern like 'r01c01-f06' if CSV is invalid, None otherwise
    """
    if not csv_path.exists():
        # Extract pattern from filename: cell-r01c01-f06-ch01.csv -> r01c01-f06
        stem = csv_path.stem  # cell-r01c01-f06-ch01
        if stem.startswith('cell-'):
            parts = stem.split('-')
            if len(parts) >= 4:  # cell, r01c01, f06, ch01
                return f"{parts[1]}-{parts[2]}"  # r01c01-f06
        return None

    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                # Extract pattern from filename: cell-r01c01-f06-ch01.csv -> r01c01-f06
                stem = csv_path.stem  # cell-r01c01-f06-ch01
                if stem.startswith('cell-'):
                    parts = stem.split('-')
                    if len(parts) >= 4:  # cell, r01c01, f06, ch01
                        return f"{parts[1]}-{parts[2]}"  # r01c01-f06
                return None
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {e}")
        # Extract pattern from filename as fallback
        stem = csv_path.stem
        if stem.startswith('cell-'):
            parts = stem.split('-')
            if len(parts) >= 4:
                return f"{parts[1]}-{parts[2]}"
        return None

    return None


def fix_csv_labels_in_file(csv_path: Path, sample_id: str, invalid_patterns: set) -> int:
    """
    Fix the Label column in a single CSV file.

    Args:
        csv_path: Path to the CSV file to fix
        sample_id: Sample identifier (e.g., 'r01c01')
        invalid_patterns: Set to collect invalid patterns

    Returns:
        Number of labels fixed
    """
    # Check if CSV is invalid and collect pattern
    invalid_pattern = collect_invalid_pattern_from_csv(csv_path, sample_id)
    if invalid_pattern:
        logger.warning(f"Invalid CSV file detected: {csv_path}, pattern: {invalid_pattern}")
        invalid_patterns.add(invalid_pattern)
        return 0

    try:
        # Read the CSV file
        rows = []
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            logger.warning(f"No data in CSV file: {csv_path}")
            return 0

        # Fix the labels
        fixed_count = 0
        for row in rows:
            if 'Label' in row:
                old_label = row['Label']

                # Skip if label is already in correct format
                if is_label_already_correct(old_label, sample_id):
                    logger.debug(f"Label already correct: {old_label}")
                    continue

                # Extract metadata from old label
                meta = extract_metadata_from_label(old_label)

                if meta['cell_id'] and meta['field'] and meta['channel']:
                    # Generate new proper filename
                    new_label = generate_proper_filename(
                        sample_id, meta['field'], meta['channel'], meta['cell_id']
                    )

                    if new_label != old_label:
                        row['Label'] = new_label
                        fixed_count += 1
                        logger.debug(f"Fixed label: {old_label} -> {new_label}")
                else:
                    logger.warning(f"Could not extract sufficient metadata from label: {old_label}")

        # Write back the fixed CSV
        if fixed_count > 0:
            with open(csv_path, 'w', newline='') as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)

            logger.info(f"Fixed {fixed_count} labels in {csv_path}")
        else:
            logger.info(f"No labels needed fixing in {csv_path}")

        return fixed_count

    except Exception as e:
        logger.error(f"Failed to fix labels in {csv_path}: {e}")
        return 0


def fix_csv_labels_for_sample(sample_id: str, invalid_patterns: set) -> int:
    """
    Fix CSV labels for all channels in a sample.

    Args:
        sample_id: Sample identifier (e.g., 'r01c01')
        invalid_patterns: Set to collect invalid patterns

    Returns:
        Total number of labels fixed
    """
    sample_dir = INTERMEDIATE_ROOT / sample_id
    stats_dir = sample_dir / "stats"

    if not stats_dir.exists():
        logger.warning(f"Stats directory does not exist: {stats_dir}")
        return 0

    total_fixed = 0

    # Process each channel
    for ch_name in CHANNEL_MAP.values():
        ch_stats_dir = stats_dir / ch_name
        if not ch_stats_dir.exists():
            logger.debug(f"Channel stats directory does not exist: {ch_stats_dir}")
            continue

        # Process all CSV files in this channel directory
        for csv_file in ch_stats_dir.glob("*.csv"):
            if csv_file.name.startswith('.'):  # Skip lock files
                continue

            fixed_count = fix_csv_labels_in_file(csv_file, sample_id, invalid_patterns)
            total_fixed += fixed_count

    return total_fixed


def save_invalid_patterns(invalid_patterns: set, output_file: Path) -> None:
    """
    Save invalid patterns to a file for cleanup.

    Args:
        invalid_patterns: Set of invalid patterns like 'r01c01-f06'
        output_file: Path to save the patterns
    """
    if not invalid_patterns:
        logger.info("No invalid patterns found")
        return

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            for pattern in sorted(invalid_patterns):
                f.write(f"{pattern}\n")

        logger.info(f"Saved {len(invalid_patterns)} invalid patterns to {output_file}")

    except Exception as e:
        logger.error(f"Failed to save invalid patterns: {e}")


def fix_all_csv_labels() -> None:
    """
    Fix CSV labels for all samples in the dataset.
    """
    logger.info("Starting CSV label correction process...")

    if not INTERMEDIATE_ROOT.exists():
        raise FileNotFoundError(f"Intermediate directory {INTERMEDIATE_ROOT} does not exist")

    total_samples = 0
    total_labels_fixed = 0
    invalid_patterns = set()

    # Process each sample directory
    for sample_dir in INTERMEDIATE_ROOT.glob("r*"):
        if not sample_dir.is_dir():
            continue

        sample_id = sample_dir.name
        logger.info(f"Fixing CSV labels for sample: {sample_id}")

        try:
            labels_fixed = fix_csv_labels_for_sample(sample_id, invalid_patterns)
            total_labels_fixed += labels_fixed
            total_samples += 1

        except Exception as e:
            logger.error(f"Failed to fix labels for sample {sample_id}: {e}")
            continue

    # Save invalid patterns for cleanup
    invalid_patterns_file = INTERMEDIATE_ROOT.parent / "invalid_patterns.txt"
    save_invalid_patterns(invalid_patterns, invalid_patterns_file)

    logger.info(f"CSV label correction completed! Fixed {total_labels_fixed} labels across {total_samples} samples.")
    if invalid_patterns:
        logger.warning(f"Found {len(invalid_patterns)} invalid patterns. Check {invalid_patterns_file} for cleanup.")


def main() -> None:
    """Main entry point for the CSV label fixing script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        fix_all_csv_labels()

    except Exception as e:
        logger.error(f"CSV label fixing failed: {e}")
        raise


if __name__ == "__main__":
    main()