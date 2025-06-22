"""
Utility for filtering data based on missing values and sample/ROI combinations.

This script:
1. Identifies rows with missing values in combined_cell_data.parquet
2. Saves incomplete rows to a separate file
3. Saves clean version without missing rows
4. Filters patches based on sample_name + roi_name combinations from parquet data
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Config
INTERMEDIATE_ROOT = Path("data/02_intermediate")
PRIMARY_ROOT = Path("data/03_primary")


def ensure_directory_exists(path: Path) -> None:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        path: Path to directory
    """
    path.mkdir(parents=True, exist_ok=True)


def get_exclusion_combinations(df: pd.DataFrame) -> Set[Tuple[str, str]]:
    """
    Extract sample_name + roi_name combinations that should be excluded from patches.

    Args:
        df: DataFrame with sample_name and roi_name columns

    Returns:
        Set of (sample_name, roi_name) tuples to exclude
    """
    exclusion_combinations = set()

    if 'sample_name' not in df.columns or 'roi_name' not in df.columns:
        logger.warning("sample_name or roi_name columns not found in data")
        return exclusion_combinations

    # Get rows with missing values - these are the ones to exclude
    incomplete_mask = df.isnull().any(axis=1)
    incomplete_df = df[incomplete_mask]

    for _, row in incomplete_df.iterrows():
        sample_name = row.get('sample_name')
        roi_name = row.get('roi_name')

        if pd.notna(sample_name) and pd.notna(roi_name):
            exclusion_combinations.add((str(sample_name), str(roi_name)))

    logger.info(f"Found {len(exclusion_combinations)} sample_name + roi_name combinations to exclude")

    return exclusion_combinations


def save_exclusion_combinations(exclusion_combinations: Set[Tuple[str, str]]) -> str:
    """
    Save exclusion combinations to a file in 03_primary directory.

    Args:
        exclusion_combinations: Set of (sample_name, roi_name) tuples

    Returns:
        Path to the saved file
    """
    ensure_directory_exists(PRIMARY_ROOT)

    exclusions_file = PRIMARY_ROOT / "excluded_sample_roi_combinations.txt"

    try:
        with open(exclusions_file, 'w') as f:
            f.write("# Excluded sample_name + roi_name combinations\n")
            f.write("# Format: sample_name,roi_name\n")
            for sample_name, roi_name in sorted(exclusion_combinations):
                f.write(f"{sample_name},{roi_name}\n")

        logger.info(f"Saved {len(exclusion_combinations)} exclusion combinations to {exclusions_file}")
        return str(exclusions_file)

    except Exception as e:
        logger.error(f"Failed to save exclusion combinations: {e}")
        return ""


def filter_combined_cell_data() -> Tuple[Dict[str, any], pd.DataFrame]:
    """
    Filter combined_cell_data.parquet by removing rows with missing values.

    Returns:
        Tuple of (filtering results dict, original DataFrame)
    """
    logger.info("Starting combined_cell_data filtering...")

    # Input and output paths
    input_file = INTERMEDIATE_ROOT / "combined_cell_data.parquet"
    output_clean = PRIMARY_ROOT / "combined_cell_data_clean.parquet"
    output_incomplete = PRIMARY_ROOT / "combined_cell_data_incomplete.parquet"

    # Ensure output directory exists
    ensure_directory_exists(PRIMARY_ROOT)

    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        return {
            "status": "failed",
            "error": f"Input file not found: {input_file}"
        }, pd.DataFrame()

    try:
        # Load the data
        logger.info(f"Loading data from {input_file}")
        df = pd.read_parquet(input_file)

        original_count = len(df)
        logger.info(f"Original dataset has {original_count} rows")

        # Identify rows with missing values
        incomplete_mask = df.isnull().any(axis=1)
        incomplete_df = df[incomplete_mask]
        clean_df = df[~incomplete_mask]

        incomplete_count = len(incomplete_df)
        clean_count = len(clean_df)

        logger.info(f"Found {incomplete_count} rows with missing values")
        logger.info(f"Clean dataset has {clean_count} rows")

        # Save incomplete rows
        if incomplete_count > 0:
            logger.info(f"Saving incomplete rows to {output_incomplete}")
            incomplete_df.to_parquet(output_incomplete, index=False)

        # Save clean data
        logger.info(f"Saving clean data to {output_clean}")
        clean_df.to_parquet(output_clean, index=False)

        # Generate summary of missing values by column
        missing_summary = df.isnull().sum()
        missing_columns = missing_summary[missing_summary > 0].to_dict()

        results = {
            "status": "success",
            "original_rows": original_count,
            "incomplete_rows": incomplete_count,
            "clean_rows": clean_count,
            "missing_by_column": missing_columns,
            "clean_file": str(output_clean),
            "incomplete_file": str(output_incomplete) if incomplete_count > 0 else None,
            "message": f"Filtered {original_count} rows: {clean_count} clean, {incomplete_count} incomplete"
        }

        return results, df

    except Exception as e:
        logger.error(f"Failed to filter combined_cell_data: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }, pd.DataFrame()


def filter_patches_by_sample_roi(exclusion_combinations: Set[Tuple[str, str]]) -> Dict[str, any]:
    """
    Filter patches from 02_intermediate to 03_primary based on sample_name + roi_name exclusions.

    Args:
        exclusion_combinations: Set of (sample_name, roi_name) tuples to exclude

    Returns:
        Dictionary with filtering results
    """
    logger.info("Starting patches filtering...")

    if not exclusion_combinations:
        logger.info("No exclusion combinations provided, copying all patches")

    # Ensure output directory exists
    ensure_directory_exists(PRIMARY_ROOT)

    copied_files = []
    excluded_files = []

    try:
        # Find all patch files in intermediate directory
        patch_extensions = [".tiff", ".tif", ".png", ".jpg", ".jpeg"]

        for sample_dir in INTERMEDIATE_ROOT.glob("r*"):
            if not sample_dir.is_dir():
                continue

            patches_dir = sample_dir / "patches"
            if not patches_dir.exists():
                continue

            logger.info(f"Processing patches from {sample_dir.name}")

            # Create corresponding directory in primary
            primary_sample_dir = PRIMARY_ROOT / sample_dir.name
            primary_patches_dir = primary_sample_dir / "patches"

            for patch_file in patches_dir.rglob("*"):
                if not patch_file.is_file():
                    continue

                if not any(patch_file.suffix.lower() == ext for ext in patch_extensions):
                    continue

                # Check if file matches any exclusion combination
                # Example filename: cell-r05c04-f01-ch01-ID0007.tiff
                # Extract sample_name (r05c04) and roi_name (ID0007)
                filename = patch_file.name
                excluded = False

                for sample_name, roi_name in exclusion_combinations:
                    if sample_name in filename and roi_name in filename:
                        excluded_files.append(str(patch_file))
                        logger.debug(f"Excluding file: {patch_file} (sample: {sample_name}, roi: {roi_name})")
                        excluded = True
                        break

                if not excluded:
                    # Calculate relative path from patches directory
                    rel_path = patch_file.relative_to(patches_dir)
                    target_file = primary_patches_dir / rel_path

                    # Ensure target directory exists
                    ensure_directory_exists(target_file.parent)

                    # Copy the file
                    shutil.copy2(patch_file, target_file)
                    copied_files.append(str(target_file))
                    logger.debug(f"Copied: {patch_file} -> {target_file}")

        logger.info(f"Copied {len(copied_files)} patch files")
        logger.info(f"Excluded {len(excluded_files)} patch files")

        return {
            "status": "success",
            "copied_files": len(copied_files),
            "excluded_files": len(excluded_files),
            "exclusion_combinations": list(exclusion_combinations),
            "copied_file_list": copied_files[:10],  # First 10 for logging
            "excluded_file_list": excluded_files[:10],  # First 10 for logging
            "message": f"Filtered patches: {len(copied_files)} copied, {len(excluded_files)} excluded"
        }

    except Exception as e:
        logger.error(f"Failed to filter patches: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }


def filter_data() -> Dict[str, any]:
    """
    Main function to filter data - both combined_cell_data and patches.
    Filters based on sample_name + roi_name combinations from incomplete rows.

    Returns:
        Dictionary with complete filtering results
    """
    logger.info("Starting data filtering process...")

    try:
        # Filter combined_cell_data and get original DataFrame
        cell_data_results, original_df = filter_combined_cell_data()

        if cell_data_results["status"] != "success":
            return cell_data_results

        # Get exclusion combinations from incomplete rows
        exclusion_combinations = get_exclusion_combinations(original_df)

        # Save exclusion combinations to file
        exclusions_file = save_exclusion_combinations(exclusion_combinations)

        # Filter patches based on exclusion combinations
        patches_results = filter_patches_by_sample_roi(exclusion_combinations)

        # Combine results
        combined_results = {
            "status": "success" if cell_data_results["status"] == "success" and patches_results["status"] == "success" else "partial_failure",
            "cell_data_filtering": cell_data_results,
            "patches_filtering": patches_results,
            "exclusions_file": exclusions_file,
            "exclusion_combinations_count": len(exclusion_combinations),
            "message": f"Data filtering completed. Cell data: {cell_data_results['status']}, Patches: {patches_results['status']}"
        }

        logger.info(f"Data filtering completed: {combined_results['message']}")
        return combined_results

    except Exception as e:
        logger.error(f"Data filtering failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }


def main() -> None:
    """Main entry point for the data filtering script."""
    parser = argparse.ArgumentParser(
        description="Filter skin cancer detection data based on missing values and sample/ROI combinations"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level"
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        results = filter_data()
        if results["status"] == "success":
            logger.info("Data filtering completed successfully!")
            logger.info(f"Results: {results['message']}")
            if results.get("exclusions_file"):
                logger.info(f"Exclusions saved to: {results['exclusions_file']}")
        else:
            logger.error(f"Data filtering failed: {results.get('error', 'Unknown error')}")
            exit(1)

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()