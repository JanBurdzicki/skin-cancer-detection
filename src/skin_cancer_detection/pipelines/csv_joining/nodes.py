"""
CSV joining pipeline nodes for skin cancer detection project.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any

from skin_cancer_detection.utils.csv_joiner import (
    join_all_csv_files,
    save_combined_data,
)

logger = logging.getLogger(__name__)


def join_csv_files_node(parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Kedro node to join all CSV files from restructured data.

    Args:
        parameters: Pipeline parameters including GFP threshold

    Returns:
        Combined DataFrame
    """
    gfp_threshold = parameters.get("gfp_threshold", 25.0)
    fix_labels = parameters.get("fix_labels", True)
    input_dir = Path(parameters.get("input_dir", "data/02_intermediate"))
    output_dir = Path(parameters.get("output_dir", "data/02_intermediate"))

    logger.info(f"Joining CSV files with GFP threshold: {gfp_threshold}")

    combined_df, channel_dfs = join_all_csv_files(
        gfp_threshold=gfp_threshold,
        fix_labels=fix_labels,
        input_root=input_dir,
        output_root=output_dir
    )

    return combined_df


def save_csv_output_node(combined_df: pd.DataFrame, parameters: Dict[str, Any]) -> str:
    """
    Kedro node to save combined data as CSV.

    Args:
        combined_df: Combined DataFrame
        parameters: Pipeline parameters

    Returns:
        Path to saved CSV file
    """
    output_dir = Path(parameters.get("output_dir", "data/02_intermediate"))
    csv_path = output_dir / "combined_cell_data.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(csv_path, index=False)

    logger.info(f"Saved combined CSV: {csv_path} ({len(combined_df)} rows)")
    return str(csv_path)


def save_parquet_output_node(combined_df: pd.DataFrame, parameters: Dict[str, Any]) -> str:
    """
    Kedro node to save combined data as Parquet.

    Args:
        combined_df: Combined DataFrame
        parameters: Pipeline parameters

    Returns:
        Path to saved Parquet file
    """
    output_dir = Path(parameters.get("output_dir", "data/02_intermediate"))
    parquet_path = output_dir / "combined_cell_data.parquet"

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(parquet_path, index=False)

    logger.info(f"Saved combined Parquet: {parquet_path} ({len(combined_df)} rows)")
    return str(parquet_path)


def create_visualization_node(combined_df: pd.DataFrame, parameters: Dict[str, Any]) -> str:
    """
    Kedro node to create GFP visualization.

    Args:
        combined_df: Combined DataFrame with GFP data
        parameters: Pipeline parameters

    Returns:
        Path to saved visualization
    """
    from skin_cancer_detection.utils.csv_joiner import create_gfp_visualization

    output_dir = Path(parameters.get("output_dir", "data/02_intermediate"))
    viz_path = output_dir / "gfp_intensity_analysis.png"

    output_dir.mkdir(parents=True, exist_ok=True)
    create_gfp_visualization(combined_df, viz_path)

    return str(viz_path)


def generate_summary_node(combined_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Kedro node to generate summary statistics of the combined data.

    Args:
        combined_df: Combined DataFrame

    Returns:
        Dictionary with summary statistics
    """
    # Check and fix any duplicate columns
    if combined_df.columns.duplicated().any():
        logger.warning("Found duplicate columns in DataFrame, removing duplicates")
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    
    summary = {
        "total_rows": len(combined_df),
        "total_columns": len(combined_df.columns),
        "unique_samples": combined_df['sample_name'].nunique() if 'sample_name' in combined_df.columns else 0,
        "unique_rois": combined_df['roi_name'].nunique() if 'roi_name' in combined_df.columns else 0,
        "columns": list(combined_df.columns)
    }

    # Add GFP status distribution if available
    if 'gfp_status' in combined_df.columns:
        try:
            # Ensure we're working with a Series, not a DataFrame
            gfp_status_series = combined_df['gfp_status']
            if hasattr(gfp_status_series, 'iloc'):
                # If it's still a DataFrame (shouldn't happen after our fix), take the first column
                if len(gfp_status_series.shape) > 1:
                    gfp_status_series = gfp_status_series.iloc[:, 0]
                    
            status_counts = gfp_status_series.value_counts()
            summary["gfp_status_distribution"] = status_counts.to_dict()
        except Exception as e:
            logger.error(f"Error processing gfp_status column: {e}")
            summary["gfp_status_error"] = str(e)

    # Add basic statistics for numeric columns
    numeric_columns = combined_df.select_dtypes(include=['number']).columns
    if len(numeric_columns) > 0:
        summary["numeric_stats"] = combined_df[numeric_columns].describe().to_dict()

    logger.info(f"Data summary: {summary['total_rows']} rows, {summary['total_columns']} columns")

    return summary