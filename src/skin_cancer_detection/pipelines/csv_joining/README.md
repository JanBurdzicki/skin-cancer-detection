# CSV Joining Pipeline

This pipeline combines all CSV files from the restructured dataset into unified CSV and Parquet files with proper labeling and visualization.

## Dependencies

This pipeline depends on:
- **csv_label_fixing**: Ensures CSV labels are properly formatted before joining
- **data_restructuring**: Provides the structured input data

## Functionality

### What it does:
1. **Label Fixing**: Runs CSV label correction if needed
2. **File Collection**: Scans all sample directories for CSV files
3. **Channel Processing**: Processes DAPI, GFP, and MT channels separately
4. **Data Merging**: Combines all channels into a single dataset
5. **GFP Labeling**: Adds positive/negative labels based on GFP intensity
6. **Visualization**: Creates comprehensive GFP analysis plots
7. **Output Generation**: Saves data in both CSV and Parquet formats

### Outputs:
- `combined_cell_data.csv` - Complete dataset in CSV format
- `combined_cell_data.parquet` - Complete dataset in Parquet format (faster loading)
- `gfp_intensity_analysis.png` - Multi-panel GFP visualization with:
  - Histogram of GFP intensities
  - Box plots by positive/negative status
  - Log-scale distribution
  - Summary statistics

## Usage

### Via Kedro Pipeline:
```bash
# Run the full pipeline (includes dependencies)
kedro run --pipeline csv_joining

# Run with custom parameters
kedro run --pipeline csv_joining --params gfp_threshold:30.0

# Run after csv_label_fixing
kedro run --pipeline csv_label_fixing,csv_joining
```

### Via Utils Script:
```bash
# Run with default settings
python -m skin_cancer_detection.utils.csv_joiner

# Run with custom threshold
python -m skin_cancer_detection.utils.csv_joiner --gfp-threshold 30.0

# Skip label fixing (if already done)
python -m skin_cancer_detection.utils.csv_joiner --skip-label-fix

# Custom output paths
python -m skin_cancer_detection.utils.csv_joiner \
  --output-csv /path/to/output.csv \
  --output-parquet /path/to/output.parquet
```

### Via Standalone Script:
```bash
python scripts/join_csv_data.py
```

## Configuration

Configure the pipeline via `conf/base/parameters/csv_joining.yml`:

```yaml
# GFP labeling parameters
gfp_threshold: 25.0  # Threshold for positive/negative classification

# Processing options
fix_labels: true  # Whether to run CSV label fixing before joining

# Input/Output paths
input_dir: "data/02_intermediate"
output_dir: "data/02_intermediate"
```

## Expected Input Structure

The pipeline expects data restructured by the `data_restructuring` pipeline:

```
data/02_intermediate/
├── r01c01/
│   └── stats/
│       ├── dapi/
│       │   └── cell-r01c01-f01-ch01-ID0001.csv
│       ├── gfp/
│       │   └── cell-r01c01-f01-ch02-ID0001.csv
│       └── mt/
│           └── cell-r01c01-f01-ch03-ID0001.csv
└── r01c02/
    └── stats/
        └── ...
```

## Output Data Schema

The combined dataset includes:
- **Metadata**: `sample_name`, `roi_name`
- **DAPI measurements**: `dapi_mean`, `dapi_std`, `dapi_min`, `dapi_max`, etc.
- **GFP measurements**: `gfp_mean`, `gfp_std`, `gfp_min`, `gfp_max`, etc.
- **MT measurements**: `mt_mean`, `mt_std`, `mt_min`, `mt_max`, etc.
- **Labels**: `gfp_status` (positive/negative based on threshold)