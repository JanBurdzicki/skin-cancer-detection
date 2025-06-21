"""
CSV file joining utilities for skin cancer detection project.

This module provides functions to join all CSV files from the restructured dataset
into unified CSV and parquet files with proper labeling and visualization.
"""

import argparse
import logging
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from .fix_csv_labels import fix_all_csv_labels

# Setup logging
logger = logging.getLogger(__name__)

# Configuration
INTERMEDIATE_ROOT = Path("data/02_intermediate")
OUTPUT_ROOT = Path("data/02_intermediate")
CHANNEL_NAMES = ["dapi", "gfp", "mt"]


def check_and_fix_csv_labels(input_root: Path) -> None:
    """
    Check if CSV labels need fixing and fix them if necessary.
    This ensures proper data consistency before joining.

    Args:
        input_root: Path to the intermediate data directory
    """
    logger.info("Checking CSV labels before joining...")

    # Check if we have sample directories
    sample_dirs = list(input_root.glob("r*c*"))
    if not sample_dirs:
        logger.warning("No sample directories found for label checking")
        return

    # For now, always run the fix to ensure consistency
    logger.info("Running CSV label correction to ensure consistency...")
    fix_all_csv_labels()


def extract_roi_and_sample_from_label(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """
    Extract roi_name and sample_name from the Label column using regex.

    Args:
        df: DataFrame with Label column containing values like 'cell-r05c04-f01-ch01-ID0056.tiff'
        csv_path: Path to CSV file (used for channel type)

    Returns:
        DataFrame with added sample_name and roi_name columns
    """
    df = df.copy()

    if 'Label' not in df.columns:
        logger.warning(f"No Label column found in {csv_path}")
        df['sample_name'] = 'unknown'
        df['roi_name'] = 'unknown'
        return df

    # Regex pattern to extract: cell-r05c04-f01-ch01-ID0056.tiff
    # Groups: (r05c04)-(f01)-(ID0056)
    pattern = r'cell-(r\d+c\d+)-(f\d+)-ch\d+-(ID\d+)(?:\.tiff?)?'

    def parse_label(label):
        """Parse a single label using regex."""
        if pd.isna(label) or not isinstance(label, str):
            return 'unknown', 'unknown'

        match = re.match(pattern, label)
        if match:
            sample_id, field, roi_id = match.groups()
            sample_name = f"{sample_id}-{field}"  # r05c04-f01
            roi_name = roi_id  # ID0056
            return sample_name, roi_name
        else:
            logger.warning(f"Label '{label}' doesn't match expected pattern")
            return 'unknown', 'unknown'

    # Apply the parsing function
    parsed_data = df['Label'].apply(parse_label)
    df['sample_name'] = [x[0] for x in parsed_data]
    df['roi_name'] = [x[1] for x in parsed_data]

    return df


def load_and_process_csv_file(csv_path: Path, channel_prefix: str) -> pd.DataFrame:
    """
    Load and process a single CSV file with proper column renaming.

    Args:
        csv_path: Path to CSV file
        channel_prefix: Prefix for column names (dapi, gfp, mt)

    Returns:
        Processed DataFrame
    """
    try:
        df = pd.read_csv(csv_path)

        # Extract roi_name and sample_name from Label column
        df = extract_roi_and_sample_from_label(df, csv_path)

        # Define columns to keep and rename
        column_mapping = {
            'Mean': f'{channel_prefix}_mean',
            'StdDev': f'{channel_prefix}_std',
            'Mode': f'{channel_prefix}_mode',
            'Min': f'{channel_prefix}_min',
            'Max': f'{channel_prefix}_max',
            'IntDen': f'{channel_prefix}_intden',
            'Median': f'{channel_prefix}_median',
            'Skew': f'{channel_prefix}_skew',
            'Kurt': f'{channel_prefix}_kurt',
            'RawIntDen': f'{channel_prefix}_rawintden',
            'Area': f'{channel_prefix}_area',
            'Perim.': f'{channel_prefix}_perimeter'
        }

        # Rename columns that exist
        existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_columns)

        # Keep metadata and renamed measurement columns
        keep_columns = ['sample_name', 'roi_name']
        keep_columns.extend(existing_columns.values())

        df = df[[col for col in keep_columns if col in df.columns]]

        return df

    except Exception as e:
        logger.error(f"Failed to process {csv_path}: {e}")
        return pd.DataFrame()


def collect_all_csv_files(input_root: Path) -> Dict[str, List[Path]]:
    """
    Collect all CSV files from the restructured data by channel.

    Args:
        input_root: Path to the intermediate data directory

    Returns:
        Dictionary mapping channel names to lists of CSV file paths
    """
    csv_files = {channel: [] for channel in CHANNEL_NAMES}

    for sample_dir in input_root.glob("r*c*"):
        if not sample_dir.is_dir():
            continue

        stats_dir = sample_dir / "stats"
        if not stats_dir.exists():
            continue

        for channel in CHANNEL_NAMES:
            channel_dir = stats_dir / channel
            if channel_dir.exists():
                csv_files[channel].extend(channel_dir.glob("*.csv"))

    for channel, files in csv_files.items():
        logger.info(f"Found {len(files)} CSV files for {channel} channel")

    return csv_files


def process_channel_data(csv_files: List[Path], channel_name: str) -> pd.DataFrame:
    """
    Process all CSV files for a specific channel.

    Args:
        csv_files: List of CSV file paths
        channel_name: Name of the channel (dapi, gfp, mt)

    Returns:
        Combined DataFrame for the channel
    """
    if not csv_files:
        logger.warning(f"No CSV files found for {channel_name} channel")
        return pd.DataFrame()

    logger.info(f"Processing {len(csv_files)} CSV files for {channel_name} channel")

    dfs = []
    for csv_file in csv_files:
        df = load_and_process_csv_file(csv_file, channel_name)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        logger.warning(f"No valid data found for {channel_name} channel")
        return pd.DataFrame()

    # Combine all files
    combined_df = pd.concat(dfs, ignore_index=True)

    # Sort by sample_name and roi_name for consistent ordering
    combined_df = combined_df.sort_values(by=['sample_name', 'roi_name']).reset_index(drop=True)

    logger.info(f"Combined {len(csv_files)} files for {channel_name}: {len(combined_df)} total rows")
    return combined_df


def add_gfp_labels(df: pd.DataFrame, threshold: float = 25.0) -> pd.DataFrame:
    """
    Add positive/negative labels based on GFP mean intensity.

    Args:
        df: DataFrame with gfp_mean column
        threshold: Threshold for positive/negative classification

    Returns:
        DataFrame with added gfp_status column
    """
    if 'gfp_mean' not in df.columns:
        logger.warning("No gfp_mean column found for labeling")
        df['gfp_status'] = 'unknown'
        return df

    df['gfp_status'] = df['gfp_mean'].apply(
        lambda x: 'positive' if pd.notna(x) and x >= threshold else 'negative'
    )

    positive_count = (df['gfp_status'] == 'positive').sum()
    negative_count = (df['gfp_status'] == 'negative').sum()

    logger.info(f"GFP labeling complete: {positive_count} positive, {negative_count} negative")

    return df


def create_gfp_visualization(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create visualization of GFP mean intensity distribution.

    Args:
        df: DataFrame with gfp_mean column
        output_path: Path to save the plot
    """
    if 'gfp_mean' not in df.columns:
        logger.warning("No gfp_mean column found for visualization")
        return

    plt.figure(figsize=(12, 8))

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Histogram
    axes[0, 0].hist(df['gfp_mean'].dropna(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('GFP Mean Intensity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('GFP Mean Intensity Distribution')
    axes[0, 0].axvline(x=25, color='red', linestyle='--', label='Threshold (25)')
    axes[0, 0].legend()

    # Box plot by status
    if 'gfp_status' in df.columns:
        try:
            # Create a clean copy of the dataframe to avoid index issues
            df_clean = df.copy().reset_index(drop=True)
            status_data = [df_clean[df_clean['gfp_status'] == status]['gfp_mean'].dropna()
                          for status in ['negative', 'positive']]
            axes[0, 1].boxplot(status_data, labels=['Negative', 'Positive'])
            axes[0, 1].set_ylabel('GFP Mean Intensity')
            axes[0, 1].set_title('GFP Intensity by Status')
        except Exception as e:
            logger.warning(f"Could not create box plot: {e}")
            axes[0, 1].text(0.5, 0.5, 'Box plot\nnot available',
                           transform=axes[0, 1].transAxes, ha='center', va='center')

    # Log scale histogram
    axes[1, 0].hist(df['gfp_mean'].dropna(), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('GFP Mean Intensity')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('GFP Mean Intensity Distribution (Log Scale)')
    axes[1, 0].axvline(x=25, color='red', linestyle='--', label='Threshold (25)')
    axes[1, 0].legend()

    # Summary statistics
    stats_text = f"""
    Count: {df['gfp_mean'].count()}
    Mean: {df['gfp_mean'].mean():.2f}
    Median: {df['gfp_mean'].median():.2f}
    Std: {df['gfp_mean'].std():.2f}
    Min: {df['gfp_mean'].min():.2f}
    Max: {df['gfp_mean'].max():.2f}
    """

    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                   verticalalignment='center', fontfamily='monospace')
    axes[1, 1].set_title('GFP Statistics')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"GFP visualization saved to {output_path}")


def join_all_csv_files(gfp_threshold: float = 25.0, fix_labels: bool = True,
                       input_root: Path = None, output_root: Path = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Join all CSV files from restructured data.

    Args:
        gfp_threshold: Threshold for GFP positive/negative classification
        fix_labels: Whether to fix CSV labels before joining
        input_root: Path to input directory (defaults to INTERMEDIATE_ROOT)
        output_root: Path to output directory (defaults to OUTPUT_ROOT)

    Returns:
        Tuple of (combined_dataframe, individual_channel_dataframes)
    """
    if input_root is None:
        input_root = INTERMEDIATE_ROOT
    if output_root is None:
        output_root = OUTPUT_ROOT

    logger.info("Starting CSV joining process...")

    # Fix CSV labels if requested
    if fix_labels:
        check_and_fix_csv_labels(input_root)

    # Collect all CSV files
    csv_files = collect_all_csv_files(input_root)

    # Process each channel
    channel_dfs = {}
    for channel in CHANNEL_NAMES:
        logger.info(f"Processing {channel} channel...")
        channel_df = process_channel_data(csv_files[channel], channel)

        if not channel_df.empty:
            channel_dfs[channel] = channel_df
            logger.info(f"Processed {channel}: {len(channel_df)} rows")

    # Combine all channels by concatenating column-wise
    if not channel_dfs:
        raise ValueError("No valid data found in any channel")

    logger.info("Combining all channels...")

    # Sort and prepare DataFrames for concatenation
    sorted_channel_dfs = []

    for i, (channel, df) in enumerate(channel_dfs.items()):
        if not df.empty:
            # Sort by the key columns
            sorted_df = df.sort_values(by=['sample_name', 'roi_name']).reset_index(drop=True)

            if i == 0:
                # Keep all columns for the first channel (metadata + measurements)
                logger.info(f"First channel ({channel}): keeping all {len(sorted_df.columns)} columns")
            else:
                # Keep only measurement columns for subsequent channels
                measurement_cols = [col for col in sorted_df.columns
                                  if col.startswith(f'{channel}_')]
                sorted_df = sorted_df[measurement_cols]
                logger.info(f"Subsequent channel ({channel}): keeping {len(measurement_cols)} measurement columns")

            sorted_channel_dfs.append(sorted_df)
            logger.info(f"Sorted {channel} channel: {len(sorted_df)} rows, {len(sorted_df.columns)} columns")

    # Concatenate all channels column-wise
    logger.info("Concatenating channels...")
    combined_df = pd.concat(sorted_channel_dfs, axis=1).reset_index(drop=True)
    logger.info(f"After concatenation: {len(combined_df)} rows, {len(combined_df.columns)} columns")

    # Add GFP labels
    logger.info("Adding GFP labels...")
    combined_df = add_gfp_labels(combined_df, gfp_threshold)

    # Remove duplicate columns (simple and efficient)
    if combined_df.columns.duplicated().any():
        duplicate_count = combined_df.columns.duplicated().sum()
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        logger.info(f"Removed {duplicate_count} duplicate columns")

    logger.info(f"Final columns: {list(combined_df.columns)}")
    logger.info(f"Successfully combined data: {len(combined_df)} rows, {len(combined_df.columns)} columns")

    return combined_df, channel_dfs


def save_combined_data(combined_df: pd.DataFrame, output_root: Path = None,
                      output_csv: Optional[str] = None,
                      output_parquet: Optional[str] = None, output_viz: Optional[str] = None) -> None:
    """
    Save combined data as CSV and parquet files.

    Args:
        combined_df: Combined DataFrame to save
        output_root: Root directory for outputs (defaults to OUTPUT_ROOT)
        output_csv: Custom path for CSV output
        output_parquet: Custom path for parquet output
        output_viz: Custom path for visualization output
    """
    if output_root is None:
        output_root = OUTPUT_ROOT

    output_root.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    csv_path = Path(output_csv) if output_csv else output_root / "combined_cell_data.csv"
    combined_df.to_csv(csv_path, index=False)
    logger.info(f"Combined CSV saved: {csv_path}")

    # Save as parquet
    parquet_path = Path(output_parquet) if output_parquet else output_root / "combined_cell_data.parquet"
    combined_df.to_parquet(parquet_path, index=False)
    logger.info(f"Combined parquet saved: {parquet_path}")

    # Create visualization
    viz_path = Path(output_viz) if output_viz else output_root / "gfp_intensity_analysis.png"
    create_gfp_visualization(combined_df, viz_path)


def main() -> None:
    """Main entry point for the CSV joining script."""
    parser = argparse.ArgumentParser(
        description="Join all CSV files from restructured skin cancer detection data"
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(INTERMEDIATE_ROOT),
        help=f"Input directory with restructured data (default: {INTERMEDIATE_ROOT})"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_ROOT),
        help=f"Output directory for combined files (default: {OUTPUT_ROOT})"
    )

    parser.add_argument(
        "--gfp-threshold",
        type=float,
        default=25.0,
        help="GFP intensity threshold for positive/negative classification (default: 25.0)"
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        help="Custom path for CSV output file"
    )

    parser.add_argument(
        "--output-parquet",
        type=str,
        help="Custom path for parquet output file"
    )

    parser.add_argument(
        "--output-viz",
        type=str,
        help="Custom path for visualization output file"
    )

    parser.add_argument(
        "--skip-label-fix",
        action="store_true",
        help="Skip CSV label fixing step"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    try:
        input_root = Path(args.input_dir)
        output_root = Path(args.output_dir)

        combined_df, channel_dfs = join_all_csv_files(
            gfp_threshold=args.gfp_threshold,
            fix_labels=not args.skip_label_fix,
            input_root=input_root,
            output_root=output_root
        )

        save_combined_data(
            combined_df,
            output_root=output_root,
            output_csv=args.output_csv,
            output_parquet=args.output_parquet,
            output_viz=args.output_viz
        )

        # Print summary
        print(f"\n=== CSV JOINING SUMMARY ===")
        print(f"Total rows: {len(combined_df)}")
        print(f"Total columns: {len(combined_df.columns)}")

        if 'sample_name' in combined_df.columns and 'roi_name' in combined_df.columns:
            unique_combinations = combined_df[['sample_name', 'roi_name']]
            print(f"Unique sample_name + roi_name combinations: {len(unique_combinations)}")

        if 'gfp_status' in combined_df.columns:
            status_counts = combined_df['gfp_status'].value_counts()
            print(f"GFP Status distribution:")
            for status, count in status_counts.items():
                print(f"  {status}: {count}")

        print(f"\nOutput files:")
        print(f"  CSV: {args.output_csv or output_root / 'combined_cell_data.csv'}")
        print(f"  Parquet: {args.output_parquet or output_root / 'combined_cell_data.parquet'}")
        print(f"  Visualization: {args.output_viz or output_root / 'gfp_intensity_analysis.png'}")

    except Exception as e:
        logger.error(f"CSV joining failed: {e}")
        raise


if __name__ == "__main__":
    main()