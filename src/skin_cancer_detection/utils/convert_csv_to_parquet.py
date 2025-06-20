#!/usr/bin/env python3
"""
Standalone script to convert CSV files to Parquet format.

This script can be used independently of the Kedro pipeline to convert
CSV files in a directory tree to Parquet format for better performance.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('csv_to_parquet_conversion.log')
        ]
    )


def convert_csv_to_parquet(input_dir: Path, output_dir: Path,
                          preserve_structure: bool = True, in_place: bool = False) -> int:
    """
    Convert all CSV files in a directory tree to Parquet format.

    Args:
        input_dir: Input directory to search for CSV files
        output_dir: Output directory for Parquet files
        preserve_structure: Whether to preserve directory structure
        in_place: If True, create Parquet files alongside CSV files in the same directories

    Returns:
        Number of files successfully converted
    """
    logger = logging.getLogger(__name__)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")

    if in_place:
        logger.info(f"Converting CSV files to Parquet in-place within {input_dir}")
    else:
        logger.info(f"Converting CSV files from {input_dir} to Parquet in {output_dir}")
        logger.info(f"Preserve structure: {preserve_structure}")
        output_dir.mkdir(parents=True, exist_ok=True)
    converted_count = 0
    failed_count = 0

    # Find all CSV files
    csv_files = list(input_dir.rglob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files to convert")

    for csv_file in csv_files:
        try:
            # Skip lock files and temporary files
            if csv_file.name.startswith('.') or csv_file.name.startswith('~'):
                logger.debug(f"Skipping temporary/lock file: {csv_file}")
                continue

            logger.debug(f"Processing: {csv_file}")

            # Read CSV
            df = pd.read_csv(csv_file)

            if in_place:
                # Create Parquet file in the same directory as the CSV file
                parquet_path = csv_file.with_suffix('.parquet')
            elif preserve_structure:
                # Create corresponding output path maintaining directory structure
                rel_path = csv_file.relative_to(input_dir)
                parquet_path = output_dir / rel_path.with_suffix('.parquet')
            else:
                # Flatten structure - all files go to output_dir root
                parquet_path = output_dir / f"{csv_file.stem}.parquet"

            # Create parent directories if needed (only for non-in-place conversions)
            if not in_place:
                parquet_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as Parquet
            df.to_parquet(parquet_path, index=False)
            converted_count += 1

            logger.debug(f"Converted: {csv_file} -> {parquet_path}")

            # Log file size comparison
            csv_size = csv_file.stat().st_size
            parquet_size = parquet_path.stat().st_size
            compression_ratio = (csv_size - parquet_size) / csv_size * 100
            logger.debug(f"Size: CSV={csv_size:,} bytes, Parquet={parquet_size:,} bytes, "
                        f"Compression: {compression_ratio:.1f}%")

        except Exception as e:
            logger.error(f"Failed to convert {csv_file}: {e}")
            failed_count += 1
            continue

    logger.info(f"Conversion completed: {converted_count} successful, {failed_count} failed")
    return converted_count


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert CSV files to Parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all CSV files in data/02_intermediate to data/03_primary
  python convert_csv_to_parquet.py data/02_intermediate data/03_primary

  # Convert with flattened structure (no subdirectories)
  python convert_csv_to_parquet.py data/02_intermediate data/03_primary --no-preserve-structure

  # Convert in-place (Parquet files created alongside CSV files)
  python convert_csv_to_parquet.py data/02_intermediate data/02_intermediate --in-place

  # Verbose output
  python convert_csv_to_parquet.py data/02_intermediate data/03_primary -v
        """
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing CSV files"
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for Parquet files"
    )

    parser.add_argument(
        "--no-preserve-structure",
        action="store_true",
        help="Don't preserve directory structure (flatten all files to output root)"
    )

    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Create Parquet files alongside CSV files in the same directories"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Convert files
        converted_count = convert_csv_to_parquet(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            preserve_structure=not args.no_preserve_structure,
            in_place=args.in_place
        )

        logger.info(f"Successfully converted {converted_count} CSV files to Parquet format")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())