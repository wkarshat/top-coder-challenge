# Output Management

This document describes the complete output management system for the Legacy Analysis System, including directory structure, file organization, logging, and best practices.

## Overview

All analysis outputs are now organized in timestamped subdirectories under the `outputs/` directory. This ensures:
- **No file conflicts** between different analysis runs
- **Easy identification** of when analyses were performed
- **Clean organization** of different types of outputs
- **Historical tracking** of analysis results

## Directory Structure

```
outputs/
├── analysis_HHMMSS/              # Main analysis system outputs
│   ├── plots/                    # Visualizations and charts
│   │   ├── correlation_heatmap.png
│   │   ├── scatter_Miles_vs_Reimb.png
│   │   └── scatter_Receipts_vs_Reimb.png
│   ├── reports/                  # JSON reports and analysis summaries
│   │   └── analysis_report.json
│   └── models/                   # Model artifacts and metrics
├── linear_regression_HHMMSS/     # Linear regression analysis outputs
│   ├── day_1_miles_vs_reimb.png # Individual day analysis plots
│   ├── day_1_receipts_vs_reimb.png
│   ├── day_X_*.png              # Additional day plots
│   └── summary_analysis.png      # Multi-panel summary visualization
└── logs/                         # Centralized system logs
    └── analysis.log
```

## Timestamp Format

- **Format**: `HHMMSS` (24-hour format)
- **Example**: `030524` = 03:05:24 (3:05:24 AM)
- **Purpose**: Ensures unique directory names for each analysis run

## Analysis Types

### 1. Main Analysis System (`analysis_HHMMSS/`)

**Command**: `cd src && python -m core.main --data ../public.csv`

**Outputs**:
- `plots/correlation_heatmap.png` - Correlation matrix visualization with significance indicators
- `plots/scatter_Miles_vs_Reimb.png` - Miles vs Reimbursement scatter plot with regression line
- `plots/scatter_Receipts_vs_Reimb.png` - Receipts vs Reimbursement scatter plot with regression line
- `reports/analysis_report.json` - Comprehensive JSON report containing:
  - Analysis results from all 5 analyzers
  - Model performance metrics (R², MSE, MAE)
  - Statistical summaries and correlations
  - Clustering results and evaluation metrics
  - Time series analysis and forecasts
- `models/` - Model artifacts and serialized objects (when applicable)

**Features**:
- 5 analysis types: correlation, statistical, advanced_stats, time_series, clustering
- 2 model types: linear, ensemble
- Comprehensive JSON report with all results

### 2. Linear Regression Analysis (`linear_regression_HHMMSS/`)

**Command**: `python linear_regression.py --csv public.csv --days 1,7,14`

**Outputs**:
- `day_X_miles_vs_reimb.png` - Miles vs Reimbursement scatter plot for specific day with:
  - Regression line and R² score
  - Correlation coefficient and p-value
  - Sample size and data distribution
- `day_X_receipts_vs_reimb.png` - Receipts vs Reimbursement scatter plot for specific day with:
  - Regression line and R² score
  - Correlation coefficient and p-value
  - Sample size and data distribution
- `summary_analysis.png` - Multi-panel summary visualization containing:
  - Data distribution by days (bar chart)
  - Miles vs Receipts colored by days (scatter plot)
  - R² values by day comparison (bar chart)
  - Reimbursement distribution by days (box plots)

**Features**:
- Day-by-day analysis with binned correlations
- Detailed statistical summaries
- Configurable day ranges

## Configuration

### Automatic Directory Creation

The system automatically creates timestamped directories using:

```python
from core.utils import create_timestamped_output_dir

# Creates: outputs/analysis_HHMMSS/
output_dir = create_timestamped_output_dir('outputs', 'analysis')
```

### Path Handling

- **From main directory**: Uses `outputs/` directly
- **From src directory**: Uses `../outputs/` to ensure consistent location
- **Automatic detection**: System detects current working directory and adjusts paths

## Benefits

### ✅ **Organization**
- All outputs grouped by analysis run
- Clear separation between different analysis types
- No scattered files in main directory

### ✅ **Traceability**
- Timestamp shows exactly when analysis was performed
- Easy to correlate outputs with analysis runs
- Historical record of all analyses

### ✅ **Scalability**
- Supports multiple concurrent analyses
- No file naming conflicts
- Easy to add new analysis types

### ✅ **Maintenance**
- Old analyses can be easily identified and archived
- Clean main directory structure
- Consistent organization across all tools

## Usage Examples

### Run Main Analysis
```bash
cd src
python -m core.main --data ../public.csv
# Creates: outputs/analysis_HHMMSS/
```

### Run Linear Regression Analysis
```bash
python linear_regression.py --csv public.csv --days all
# Creates: outputs/linear_regression_HHMMSS/
```

### Custom Output Directory
```bash
python linear_regression.py --csv public.csv --output-dir custom_analysis
# Creates: custom_analysis/ (no timestamp)
```

## Migration from Old Structure

The old structure with scattered PNG files and mixed directories has been cleaned up:

**Before**:
```
├── day_1_miles_vs_reimb.png     # Scattered in main directory
├── day_1_receipts_vs_reimb.png  # Scattered in main directory
├── summary_analysis.png         # Scattered in main directory
├── output/                      # Mixed with outputs/
└── outputs/
    ├── plots/                   # Not timestamped
    ├── reports/                 # Not timestamped
    └── models/                  # Not timestamped
```

**After**:
```
outputs/
├── analysis_030524/
│   ├── plots/
│   ├── reports/
│   └── models/
└── linear_regression_030245/
    ├── day_*.png
    └── summary_analysis.png
```

## Best Practices

1. **Always use timestamped directories** for new analyses
2. **Archive old analyses** periodically to save space
3. **Use descriptive prefixes** for different analysis types
4. **Document analysis parameters** in the output directory if needed
5. **Keep the main directory clean** - no direct output files

## Logging System

### Log Configuration

Logs are centralized in `outputs/logs/` with the following structure:

```
outputs/logs/
├── analysis.log          # Main system logs
└── [other_logs]          # Additional log files as needed
```

### Log Configuration (config.yaml)

```yaml
logging:
  level: INFO
  format: '%(asctime)s - %(levelname)s - %(message)s'
  file: outputs/logs/analysis.log
  console: true
```

### Log Content

- **Data loading and validation**: Record counts, column information
- **Analysis execution**: Start/completion times, success/failure status
- **Model training**: Performance metrics, training progress
- **Output generation**: File creation, path information
- **Error handling**: Detailed error messages and stack traces

## File Management

### Automatic Cleanup

The system maintains clean organization by:
- **Timestamped directories**: Prevent file conflicts
- **Centralized outputs**: All files in designated locations
- **No scattered files**: Main directory stays clean

### Archive Strategy

For long-term maintenance:
1. **Keep recent analyses**: Last 5-10 timestamped directories
2. **Archive older results**: Move to archive directory or external storage
3. **Document significant analyses**: Add README files for important results

## Integration with Analysis Tools

### Main Analysis System
- **Configuration**: `config.yaml` and `analysis.yaml`
- **Output location**: `outputs/analysis_HHMMSS/`
- **Logging**: Comprehensive pipeline logging

### Linear Regression Tool
- **Configuration**: Command-line arguments
- **Output location**: `outputs/linear_regression_HHMMSS/`
- **Logging**: Console output with detailed statistics

### Script Interface
- **Configuration**: YAML files + CLI arguments
- **Output location**: Configurable with defaults
- **Logging**: Combined system and script logging

This comprehensive output management system ensures consistent, organized, and maintainable output handling across the entire Legacy Analysis System. 