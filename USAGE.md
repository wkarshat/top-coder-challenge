# System Usage Guide

## Quick Start

### **Main Analysis System**
```bash
cd src
python -m core.main --data ../public.csv
```
**Output**: `outputs/analysis_HHMMSS/` with plots, reports, and model results

### **Linear Regression Analysis**
```bash
python linear_regression.py --csv public.csv --days all
```
**Output**: `outputs/linear_regression_HHMMSS/` with day-by-day analysis

### **Script Interface**
```bash
python scripts/run_analysis.py --data-source public.csv
```
**Output**: Same as main system with additional CLI features

## Configuration

### **System Configuration (`config.yaml`)**
```yaml
logging:
  level: INFO
  file: outputs/logs/analysis.log
performance:
  parallel_processing: true
  max_workers: 4
```

### **Analysis Configuration (`analysis.yaml`)**
```yaml
data:
  target_column: Reimb
  feature_columns: [Days, Miles, Receipts]
analysis:
  types: [correlation, statistical, advanced_stats, time_series, clustering]
models:
  types: [linear, ensemble]
```

## Execution Methods

### **1. Core System (Recommended)**
```bash
cd src
python -m core.main --data ../public.csv
```
**Features**:
- 5 analysis types (correlation, statistical, advanced_stats, time_series, clustering)
- 2 model types (linear, ensemble)
- Automatic timestamped outputs
- Comprehensive JSON reports

### **2. Linear Regression Tool**
```bash
python linear_regression.py --csv public.csv --days 1,7,14
```
**Features**:
- Day-by-day binned correlation analysis
- Detailed statistical summaries
- Custom day ranges (`--days all`, `--days 1-5`, `--days 1,3,5`)
- Individual day plots and summary visualization

### **3. Script Interface**
```bash
python scripts/run_analysis.py --data-source public.csv
```
**Additional Features**:
- Input validation
- Batch processing
- Custom configuration files
- Verbose logging options

## Command Line Options

### **Linear Regression Script**
```bash
python linear_regression.py [OPTIONS]

Options:
  --csv, -c TEXT          Input CSV file (default: public.csv)
  --days, -d TEXT         Days to process: "all", "1-5", or "1,3,5" (default: all)
  --no-plots             Skip individual day plots
  --output-dir TEXT      Custom output directory
```

### **Script Interface**
```bash
python scripts/run_analysis.py [OPTIONS]

Options:
  --data-source TEXT     Input data file(s)
  --config TEXT          System configuration file (default: config.yaml)
  --analysis-config TEXT Analysis configuration file (default: analysis.yaml)
  --batch                Enable batch processing
  --verbose              Enable verbose logging
```

## Output Organization

All outputs are automatically organized in timestamped directories under `outputs/`. **See OUTPUT.md for complete details.**

## System Components

### **Active Analyzers**
1. **Correlation**: Pearson correlations with significance testing
2. **Statistical**: Descriptive statistics and distributions
3. **Advanced Stats**: Normality tests, outlier detection
4. **Time Series**: Trend analysis and forecasting
5. **Clustering**: K-means, DBSCAN, hierarchical clustering

### **Models**
1. **Linear**: Ridge/Lasso regression with cross-validation
2. **Ensemble**: Voting regressor with multiple base models

### **Performance**
- **Analysis Speed**: ~6 seconds for 1,000 records
- **Model Performance**: Linear R² = 0.784, Ensemble R² = 0.913
- **Parallel Processing**: Configurable via `config.yaml`

## Common Use Cases

### **Complete Analysis**
```bash
cd src
python -m core.main --data ../public.csv
```
**Result**: Full analysis with 5 analyzers, 2 models, visualizations, and JSON report

### **Day-Specific Analysis**
```bash
python linear_regression.py --csv public.csv --days 1,7,14
```
**Result**: Detailed binned correlation analysis for specific days

### **Batch Processing**
```bash
python scripts/run_analysis.py --data-source file1.csv,file2.csv --batch
```
**Result**: Analysis of multiple files with separate output directories

### **Custom Configuration**
```bash
python scripts/run_analysis.py \
    --data-source data.csv \
    --config custom_config.yaml \
    --analysis-config custom_analysis.yaml
```

## Troubleshooting

### **Common Issues**
1. **Import Errors**: Ensure you're in the correct directory (`cd src` for core system)
2. **File Not Found**: Check data file paths (use `../public.csv` from src directory)
3. **Permission Errors**: Ensure write permissions for `outputs/` directory

### **Logging**
Check `outputs/logs/analysis.log` for detailed execution information and error messages.

### **Performance**
- For large datasets, enable parallel processing in `config.yaml`
- Use `--no-plots` flag for faster linear regression analysis
- Monitor memory usage for datasets > 10,000 records

## Migration Notes

### **From Legacy System**
- Old scattered PNG files have been moved to timestamped directories
- Log files migrated from `logs/` to `outputs/logs/`
- Configuration consolidated into two YAML files

### **Removed Components**
- `BinnedAnalyzer`: Functionality available in `linear_regression.py`
- Legacy shell scripts: Not used by current system
- Scattered output files: Now organized in timestamped directories

This system provides comprehensive data analysis capabilities with clean organization and flexible execution options. 