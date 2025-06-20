"""
Data restructuring utilities for skin cancer detection project.

This module provides functions to restructure image datasets from scattered raw files
into a clean, deep-learning-ready layout.
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Optional

# Setup logging
logger = logging.getLogger(__name__)

# Config
INPUT_ROOT = Path("data/01_raw")
OUTPUT_ROOT = Path("data/02_intermediate")
CHANNEL_MAP = {
    "ch01": "dapi",
    "ch02": "gfp",
    "ch03": "mt",
    # ch04 is skipped as it's useless
}


def extract_metadata(filename: str) -> Dict[str, str]:
    """
    Extract metadata from a filename supporting two formats:
    1. "dapi_0001 - ID 0001_r01c01f02p03-ch01t01.tiff" (with ID)
    2. "r01c01f01p03-ch01t01.tiff" (without ID)

    Args:
        filename: Filename to parse

    Returns:
        Dictionary with keys: row_col, field, channel, cell_id (if present)
        (pZZ, tTT are not useful and can be ignored)

    Example:
        >>> extract_metadata("dapi_0001 - ID 0001_r01c01f02p03-ch01t01.tiff")
        {'row_col': 'r01c01', 'field': 'f02', 'channel': 'ch01', 'cell_id': 'ID0001'}
        >>> extract_metadata("r01c01f01p03-ch01t01.tiff")
        {'row_col': 'r01c01', 'field': 'f01', 'channel': 'ch01', 'cell_id': None}
    """
    try:
        cell_id = None

        # Check if this is format 1 (with ID)
        if " - ID " in filename:
            # Extract cell ID first
            id_part = filename.split(" - ID ")[1]
            cell_id = f"ID{id_part.split('_')[0].zfill(4)}"

            # Get the part after the ID for further parsing
            main_part = id_part.split('_', 1)[1]  # Everything after "ID XXXX_"
        else:
            # Format 2 - no ID, use filename directly
            main_part = filename

        # Parse the main part: rXXcXXfYYpZZ-chNNtTT.tiff
        parts = main_part.split("-")[0].split("f")
        row_col = parts[0][:6]  # rXXcXX
        field = f"f{parts[1][:2]}"  # fYY

        # Extract channel part
        ch_part = main_part.split("-")[-1]
        channel = ch_part.split("t")[0]  # chNN

        return {
            "row_col": row_col,
            "field": field,
            "channel": channel,
            "cell_id": cell_id
        }
    except (IndexError, ValueError) as e:
        logger.warning(f"Failed to extract metadata from filename {filename}: {e}")
        return {"row_col": "", "field": "", "channel": "", "cell_id": None}


def extract_cell_id(stem: str) -> Optional[str]:
    """
    Extract cell ID (e.g. ID0001) from a filename.

    Args:
        stem: Filename stem to parse

    Returns:
        Formatted cell ID or None if not found

    Example:
        >>> extract_cell_id("some_file_ID 123_other")
        'ID0123'
    """
    if "ID " in stem:
        try:
            return f"ID{stem.split('ID ')[1].split('_')[0].zfill(4)}"
        except (IndexError, ValueError) as e:
            logger.warning(f"Failed to extract cell ID from {stem}: {e}")
            return None
    return None


def write_csv(path: Path, rows: List[Dict]) -> None:
    """
    Write a CSV file from list of dictionary rows.

    Args:
        path: Output file path
        rows: List of dictionaries to write as CSV rows
    """
    if not rows:
        logger.warning(f"No data to write to {path}")
        return

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Successfully wrote {len(rows)} rows to {path}")
    except Exception as e:
        logger.error(f"Failed to write CSV to {path}: {e}")
        raise


def create_output_directories(sample_id: str) -> Dict[str, Path]:
    """
    Create output directory structure for a sample.

    Args:
        sample_id: Sample identifier (e.g., 'r01c01')

    Returns:
        Dictionary mapping directory types to Path objects
    """
    out_sample_dir = OUTPUT_ROOT / sample_id
    directories = {
        'sample': out_sample_dir,
        'images': out_sample_dir / "images",
        'patches': out_sample_dir / "patches",
        'stats': out_sample_dir / "stats"
    }

    # Create base directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # Create channel-specific subdirectories
    for ch_name in CHANNEL_MAP.values():
        (directories['patches'] / ch_name).mkdir(parents=True, exist_ok=True)
        (directories['stats'] / ch_name).mkdir(parents=True, exist_ok=True)

    return directories


def process_raw_images(sample_dir: Path, sample_id: str, out_image_dir: Path) -> None:
    """
    Process raw image fields and copy them to output directory.

    Args:
        sample_dir: Input sample directory
        sample_id: Sample identifier
        out_image_dir: Output images directory
    """
    processed_count = 0

    for ch_folder in sample_dir.glob("ch*"):
        ch_id = ch_folder.name
        if ch_id not in CHANNEL_MAP:
            if ch_id == "ch04":
                logger.debug(f"Skipping ch04 (useless channel) in {sample_dir}")
            else:
                logger.warning(f"Unknown channel {ch_id} in {sample_dir}")
            continue

        for img_file in ch_folder.glob("*.tiff"):
            try:
                meta = extract_metadata(img_file.name)
                new_image_name = f"{sample_id}-{meta['field']}-{meta['channel']}.tiff"
                out_path = out_image_dir / new_image_name
                out_path.write_bytes(img_file.read_bytes())
                processed_count += 1
            except Exception as e:
                logger.error(f"Failed to process image {img_file}: {e}")

    logger.info(f"Processed {processed_count} raw images for sample {sample_id}")


def process_patches_and_stats(sample_dir: Path, sample_id: str,
                            out_patch_dir: Path, out_stats_dir: Path) -> None:
    """
    Process patches and corresponding CSV stats.

    Args:
        sample_dir: Input sample directory
        sample_id: Sample identifier
        out_patch_dir: Output patches directory
        out_stats_dir: Output stats directory
    """
    processed_patches_count = 0
    processed_stats_count = 0

    for ch_id, ch_name in CHANNEL_MAP.items():
        # Process patches first
        patch_dir = sample_dir / f"{ch_name}_img"
        if patch_dir.exists():
            for patch_file in patch_dir.glob(f"*{ch_id}*.tiff"):
                try:
                    meta = extract_metadata(patch_file.name)

                    # Use cell_id from metadata if available, otherwise try to extract from stem
                    cell_id = meta.get('cell_id') or extract_cell_id(patch_file.stem)

                    if not cell_id:
                        logger.warning(f"No cell ID found for patch {patch_file}")
                        continue

                    # Include ID in patch name
                    patch_name = f"cell-{sample_id}-{meta['field']}-{meta['channel']}-{cell_id}.tiff"
                    patch_out_path = out_patch_dir / ch_name / patch_name
                    patch_out_path.write_bytes(patch_file.read_bytes())
                    processed_patches_count += 1

                except Exception as e:
                    logger.error(f"Failed to process patch {patch_file}: {e}")
                    continue

        # Process CSV stats
        csv_src_dir = INPUT_ROOT / f"{ch_name}_csv"
        if csv_src_dir.exists():
            for csv_file in csv_src_dir.glob(f"*{sample_id}*{ch_id}*.csv"):
                try:
                    meta = extract_metadata(csv_file.name)

                    csv_name = f"cell-{sample_id}-{meta['field']}-{meta['channel']}.csv"
                    csv_out_path = out_stats_dir / ch_name / csv_name
                    csv_out_path.write_bytes(csv_file.read_bytes())
                    processed_stats_count += 1

                except Exception as e:
                    logger.error(f"Failed to process CSV {csv_file}: {e}")
                    continue

    logger.info(f"Processed {processed_patches_count} patches and {processed_stats_count} stats for sample {sample_id}")


def restructure_data() -> None:
    """
    Main function to restructure image dataset from scattered raw files into
    a clean, deep-learning-ready layout.
    """
    logger.info("Starting data restructuring process...")

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input directory {INPUT_ROOT} does not exist")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    processed_samples = 0

    for sample_dir in INPUT_ROOT.glob("r*"):
        if not sample_dir.is_dir():
            continue

        sample_id = sample_dir.name  # e.g. r01c01
        logger.info(f"Processing sample: {sample_id}")

        try:
            # Create output directories
            directories = create_output_directories(sample_id)

            # Process raw images
            process_raw_images(sample_dir, sample_id, directories['images'])

            # Process patches and stats
            process_patches_and_stats(
                sample_dir, sample_id, directories['patches'], directories['stats']
            )

            processed_samples += 1

        except Exception as e:
            logger.error(f"Failed to process sample {sample_id}: {e}")
            continue

    logger.info(f"Data restructuring completed successfully! Processed {processed_samples} samples.")


def main() -> None:
    """Main entry point for the restructuring script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        restructure_data()

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
