# Data Cleanup Pipeline

This pipeline cleans up invalid images and patches based on patterns identified during CSV label fixing.

## Overview

When the CSV label fixing process encounters empty or missing CSV files, it identifies invalid patterns (e.g., "r01c01-f06") that indicate failed image analysis. This pipeline removes all related files containing those patterns.

## Pipeline Structure

### Nodes

1. **cleanup_invalid_data_dry_run_node**: Performs a dry run to show what files would be removed
2. **cleanup_invalid_data_execute_node**: Actually removes the identified files

### Dependencies

This pipeline depends on the `csv_label_fixing` pipeline to generate the `invalid_patterns.txt` file.

## Usage

### 1. Dry Run (Recommended First)

To see what files would be removed without actually deleting them:

```bash
kedro run --pipeline data_cleanup --tags dry_run
```

### 2. Execute Cleanup

To actually remove the invalid files:

```bash
kedro run --pipeline data_cleanup --tags execute
```

### 3. Standalone Script

You can also use the cleanup utility directly:

```bash
# Dry run
python -m skin_cancer_detection.utils.cleanup_invalid_data --dry-run

# Execute
python -m skin_cancer_detection.utils.cleanup_invalid_data --execute
```

## What Gets Removed

The cleanup process removes files containing invalid patterns from:

- **Raw data directory** (`01_raw/`): Original TIFF images
- **Intermediate data directory** (`02_intermediate/`):
  - Restructured TIFF images
  - Generated patches (PNG, JPG)
  - Empty or invalid CSV files

### Example

If pattern "r01c01-f06" is invalid, the following files would be removed:
- All files containing "r01c01-f06" in their filename
- Examples:
  - `dapi_r01c01f06p03-ch01t01.tiff`
  - `cell-r01c01-f06-ch01-ID0001.tiff`
  - `patches/cell-r01c01-f06-ch01-ID0001.png`

## Configuration

Configuration is in `conf/base/parameters/data_cleanup.yml`:

```yaml
# Path to the file containing invalid patterns
invalid_patterns_file: "data/invalid_patterns.txt"
```

## Safety Features

1. **Dry run by default**: Always shows what would be removed first
2. **Pattern validation**: Only removes files matching specific invalid patterns
3. **Logging**: Comprehensive logging of all operations
4. **Empty directory cleanup**: Removes empty directories after file deletion

## Integration with Other Pipelines

This pipeline should be run after `csv_label_fixing` but before `csv_joining`:

```bash
# Full workflow
kedro run --pipeline csv_label_fixing
kedro run --pipeline data_cleanup --tags dry_run  # Review what will be removed
kedro run --pipeline data_cleanup --tags execute  # Actually remove files
kedro run --pipeline csv_joining
```