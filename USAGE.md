# Three Variable Fitting 809 Data - System Usage Guide

## Documentation References

- **System Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md) for complete system design, components, and technical details
- **Output Management**: See [OUTPUT.md](OUTPUT.md) for comprehensive output structure, file formats, and directory organization
- **Project Requirements**: See [PRD.md](PRD.md) for original project requirements and specifications

## Quick Start

### **Main Analysis System**
```bash
python src/core/main.py --csv public.csv --config config.yaml --analysis-config analysis.yaml
```
**Output**: `outputs/main_20250608/run001/` with standardized results, visualizations, and metadata

### **Comprehensive Binned Analysis**
```bash
python -m src.analysis.binned.comprehensive_analysis --csv public.csv
```
**Output**: `outputs/comprehensive_r8_20250608/run001/` with 8-region analysis

### **Script Interface**
```bash
python scripts/run_analysis.py --csv public.csv
```
**Output**: `outputs/script_20250608/run001/` with identical functionality to main system

## All Analysis Programs

### **1. Main System Analysis**
```bash
python src/core/main.py --csv public.csv --config config.yaml --analysis-config analysis.yaml
```
- **Directory**: `outputs/main_20250608/`
- **Performance**: Linear R² = 0.784, Ensemble R² = 0.913
- **Features**: 5 analyzers, 2 models, comprehensive JSON reports
- **Files**: `results.json`, `summary.txt`, correlation plots, scatter plots

**Command Line Options**:
- `--csv CSV` (required): Path to CSV data file to analyze
- `--config CONFIG`: Path to system configuration file (default: config.yaml)
- `--analysis-config ANALYSIS_CONFIG`: Path to analysis configuration file (default: analysis.yaml)
- `--batch`: Run batch analysis on multiple files
- `--output-dir OUTPUT_DIR`: Output directory (auto-generated if not specified)
- `--legacy`: Use legacy timestamped directory format
- `-h, --help`: Show help message and exit

### **2. Script Wrapper**
```bash
python scripts/run_analysis.py --csv public.csv
```
- **Directory**: `outputs/script_20250608/`
- **Performance**: Identical to main system
- **Features**: CLI wrapper for main orchestrator
- **Files**: Same as main system

**Command Line Options**:
- `--csv CSV` (required): CSV data file. For batch processing, separate with commas
- `--config CONFIG, -c CONFIG`: System configuration file (default: config.yaml)
- `--analysis-config ANALYSIS_CONFIG, -a ANALYSIS_CONFIG`: Analysis configuration file (default: analysis.yaml)
- `--output-dir OUTPUT_DIR, -o OUTPUT_DIR`: Output directory (auto-generated if not specified)
- `--batch, -b`: Process multiple data sources (comma-separated)
- `--verbose, -v`: Enable verbose logging
- `--quiet, -q`: Suppress output except errors
- `--legacy`: Use legacy timestamped directory format
- `-h, --help`: Show help message and exit

**Batch Processing Example**:
```bash
python scripts/run_analysis.py --csv public.csv,private.csv --batch --verbose
```

### **3. Comprehensive Binned Analysis**
```bash
python -m src.analysis.binned.comprehensive_analysis --csv public.csv
```
- **Directory**: `outputs/comprehensive_r8_20250608/`
- **Performance**: Linear R² = 0.784, Polynomial avg R² = 0.852
- **Features**: 8-region analysis, linear formula fitting, polynomial enhancement
- **Files**: `results.json`, `summary.txt`, `dashboard.png`

**Command Line Options**:
- `--csv CSV`: Input CSV file (default: public.csv)
- `--output-dir OUTPUT_DIR`: Output directory (auto-generated if not specified)
- `--max-regions MAX_REGIONS`: Maximum regions to analyze (default: 8)
- `--legacy`: Use legacy timestamped directory format
- `-h, --help`: Show help message and exit

**Custom Region Analysis**:
```bash
python -m src.analysis.binned.comprehensive_analysis --csv data.csv --max-regions 4 --legacy
```

### **4. Daily Analysis**
```bash
python -m src.analysis.binned.daily_analysis --csv public.csv
```
- **Directory**: `outputs/daily_m14_20250608/`
- **Performance**: R² range 0.585-0.872 across 14 days
- **Features**: Day-by-day analysis with performance ranking
- **Files**: `results.json`, `summary.txt`, `dashboard.png`

**Command Line Options**:
- `--csv CSV`: Input CSV file (default: public.csv)
- `--output-dir OUTPUT_DIR`: Output directory (auto-generated if not specified)
- `--max-days MAX_DAYS`: Maximum days to analyze (default: 14)
- `--legacy`: Use legacy timestamped directory format
- `-h, --help`: Show help message and exit

**Limited Day Analysis**:
```bash
python -m src.analysis.binned.daily_analysis --csv data.csv --max-days 10 --output-dir daily_results/
```

### **5. Daily Rate Analysis**
```bash
python -m src.analysis.binned.daily_rate_analysis --csv public.csv
```
- **Directory**: `outputs/daily_rate_m150_r100_20250608/`
- **Performance**: 42/56 subregions populated, R² range 0.090-0.951
- **Features**: Subregion analysis with miles/receipts thresholds
- **Files**: `results.json`, `summary.txt`, `dashboard.png`

**Command Line Options**:
- `--csv CSV`: Input CSV file (default: public.csv)
- `--output-dir OUTPUT_DIR`: Output directory (auto-generated if not specified)
- `--miles-threshold MILES_THRESHOLD`: Miles threshold (default: 150)
- `--receipts-threshold RECEIPTS_THRESHOLD`: Receipts threshold (default: 100)
- `--legacy`: Use legacy timestamped directory format
- `-h, --help`: Show help message and exit

**Custom Threshold Analysis**:
```bash
python -m src.analysis.binned.daily_rate_analysis --csv data.csv --miles-threshold 200 --receipts-threshold 150
```

### **6. Reimbursement Threshold Analysis**
```bash
python -m src.analysis.binned.reimb_threshold_analysis --csv public.csv
```
- **Directory**: `outputs/reimb_threshold_t1000_20250608/`
- **Performance**: Below $1000: R² = 0.612, Above $1000: R² = 0.619
- **Features**: Above/below threshold analysis with statistical testing
- **Files**: `results.json`, `summary.txt`, `dashboard.png`

**Command Line Options**:
- `--csv CSV`: Input CSV file (default: public.csv)
- `--output-dir OUTPUT_DIR`: Output directory (auto-generated if not specified)
- `--threshold THRESHOLD`: Reimbursement threshold (default: 1000)
- `--legacy`: Use legacy timestamped directory format
- `-h, --help`: Show help message and exit

**Custom Threshold Analysis**:
```bash
python -m src.analysis.binned.reimb_threshold_analysis --csv data.csv --threshold 1500 --legacy
```

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

## Standardized Output System

### **Directory Structure**
All outputs use standardized naming with parameter encoding:
- **Format**: `{analysis_type}_{parameters}_{series_id}/`
- **Runs**: Auto-incrementing `run001/`, `run002/`, etc.
- **Files**: Standardized `results.json`, `summary.txt`, `dashboard.png`
- **Metadata**: Complete run history in `metadata.json`

**For complete output details, see [OUTPUT.md](OUTPUT.md)**

### **Example Structure**
```
outputs/
├── comprehensive_r8_20250608/
│   ├── metadata.json
│   ├── run001/
│   │   ├── results.json
│   │   ├── summary.txt
│   │   └── dashboard.png
│   └── run002/
├── daily_m14_20250608/
├── main_20250608/
└── logs/
    └── analysis.log
```

## Command Line Options

### **Legacy Format Support**
```bash
# Use legacy timestamped format
python -m src.analysis.binned.comprehensive_analysis --csv public.csv --legacy
# Creates: outputs/comprehensive_205818/
```

### **Multiple Runs**
```bash
# First run
python -m src.analysis.binned.comprehensive_analysis --csv public.csv
# Creates: outputs/comprehensive_r8_20250608/run001/

# Second run
python -m src.analysis.binned.comprehensive_analysis --csv public.csv
# Creates: outputs/comprehensive_r8_20250608/run002/
```

## System Components

**For detailed component architecture, see [ARCHITECTURE.md](ARCHITECTURE.md)**

### **Active Analyzers**
1. **Correlation**: Pearson correlations with significance testing
2. **Statistical**: Descriptive statistics and distributions
3. **Advanced Stats**: Normality tests, outlier detection
4. **Time Series**: Trend analysis and forecasting
5. **Clustering**: K-means, DBSCAN, hierarchical clustering

### **Models**
1. **Linear**: Ridge/Lasso regression with cross-validation
2. **Ensemble**: Voting regressor with multiple base models

### **Performance Summary**
- **Analysis Speed**: ~6 seconds for 1,000 records
- **Main System**: Linear R² = 0.784, Ensemble R² = 0.913
- **Best Regional**: R² = 0.893 (Region 2)
- **Best Subregion**: R² = 0.951 (Day2_M≤300_R≤$200)

## Common Use Cases

### **Complete Analysis**
```bash
python src/core/main.py --csv public.csv --config config.yaml --analysis-config analysis.yaml
```
**Result**: Full analysis with 5 analyzers, 2 models, visualizations, and JSON report

### **Regional Pattern Analysis**
```bash
python -m src.analysis.binned.comprehensive_analysis --csv public.csv
```
**Result**: 8-region analysis with linear formula fitting and polynomial enhancement

### **Daily Performance Tracking**
```bash
python -m src.analysis.binned.daily_analysis --csv public.csv
```
**Result**: Day-by-day performance analysis with ranking

### **Threshold Comparison**
```bash
python -m src.analysis.binned.reimb_threshold_analysis --csv public.csv
```
**Result**: Statistical comparison above/below $1000 threshold

### **Subregion Analysis**
```bash
python -m src.analysis.binned.daily_rate_analysis --csv public.csv
```
**Result**: 56 subregion analysis with rate thresholds

## Analysis Results Interpretation

### **Linear Formula Results**
```
Region R2: 105.950*Days + 0.301*Miles + 0.812*Receipts + (-79.1)
R² = 0.893, RMSE = 125.4
```
- **A coefficient (Days)**: Cost per day
- **B coefficient (Miles)**: Cost per mile  
- **C coefficient (Receipts)**: Receipt multiplier
- **D coefficient (Intercept)**: Base cost
- **R²**: Variance explained (0-1, higher better)
- **RMSE**: Prediction error (lower better)

### **Performance Ranking**
1. **Region R2** (Low Days, High Miles, Low Receipts): R² = 0.893
2. **Region R0** (Low Days, Low Miles, Low Receipts): R² = 0.876
3. **Region R4** (High Days, Low Miles, Low Receipts): R² = 0.860

### **Daily Analysis Results**
- **Best Day**: Day 2 (R² = 0.872)
- **Worst Day**: Day 14 (R² = 0.585)
- **Average**: R² = 0.714 across all days

### **Subregion Analysis Results**
- **Best Subregion**: Day2_M≤300_R≤$200 (R² = 0.951)
- **Populated Subregions**: 42/56 total
- **Performance Range**: R² 0.090-0.951

## Troubleshooting

### **Common Issues**
1. **Import Errors**: Use module syntax `python -m src.analysis.binned.comprehensive_analysis`
2. **File Not Found**: Check data file paths and current directory
3. **Permission Errors**: Ensure write permissions for `outputs/` directory

### **PowerShell Syntax**
```powershell
# Correct PowerShell syntax for Windows
python -m src.analysis.binned.comprehensive_analysis --csv public.csv

# Multiple commands
python src/core/main.py --csv public.csv --config config.yaml --analysis-config analysis.yaml
python -m src.analysis.binned.daily_analysis --csv public.csv
```

### **Logging**
Check `outputs/logs/analysis.log` for detailed execution information:
```
2025-06-08 21:09:10,815 - INFO - Starting analysis pipeline for public.csv
2025-06-08 21:09:13,848 - INFO - linear model completed
2025-06-08 21:09:15,304 - INFO - Analysis pipeline completed successfully
```

### **Performance**
- For large datasets, enable parallel processing in `config.yaml`
- Monitor memory usage for datasets > 10,000 records
- Use `--legacy` flag for simple timestamped outputs

## Output File Details

**For comprehensive output documentation, see [OUTPUT.md](OUTPUT.md)**

### **Standardized Files**
- **`results.json`**: Complete analysis results with metrics and coefficients
- **`summary.txt`**: Human-readable summary with key findings
- **`dashboard.png`**: Primary visualization dashboard
- **`metadata.json`**: Run tracking with timestamps and parameters

### **File Sizes (Typical)**
- **Main System**: `results.json` (535KB), 3 visualization PNGs (30-105KB each)
- **Comprehensive**: `results.json` (48KB), `dashboard.png` (411KB)
- **Daily Analysis**: `results.json` (68KB), `dashboard.png` (437KB)
- **Daily Rate**: `results.json` (65KB), `dashboard.png` (544KB)
- **Threshold**: `results.json` (21KB), `dashboard.png` (383KB)

## Advanced Usage

### **Programmatic Access**
```python
from src.core.main import AnalysisOrchestrator

# Initialize orchestrator
orchestrator = AnalysisOrchestrator('config.yaml', 'analysis.yaml', 'outputs/main_20250608/run001')

# Run analysis
results = orchestrator.run_analysis('public.csv')

# Access results
print(f"Linear R²: {results['models']['linear']['metrics']['r2_score']}")
print(f"Ensemble R²: {results['models']['ensemble']['metrics']['r2_score']}")
```

### **Batch Analysis**
```bash
# Run all analyses
python src/core/main.py --csv public.csv --config config.yaml --analysis-config analysis.yaml
python scripts/run_analysis.py --csv public.csv
python -m src.analysis.binned.comprehensive_analysis --csv public.csv
python -m src.analysis.binned.daily_analysis --csv public.csv
python -m src.analysis.binned.daily_rate_analysis --csv public.csv
python -m src.analysis.binned.reimb_threshold_analysis --csv public.csv
```

## Legacy Reimbursement Formula

The system implements and validates the legacy reimbursement calculation:
**Reimb = Days × 100 + Miles × 0.5 + Receipts**

Where:
- **Days**: Trip duration (1-14 days in dataset)
- **Miles**: Total miles traveled (1-1400 in dataset)
- **Receipts**: Total receipt amount in dollars (1-2600 in dataset)

### **Formula Validation Results**
- **Overall Formula**: 50.050*Days + 0.446*Miles + 0.383*Receipts + 266.7 (R² = 0.784)
- **Regional Variations**: Coefficients vary significantly across 8 regions
- **Best Regional Fit**: Region R2 with R² = 0.893

## Migration Notes

### **From Legacy System**
- **Standardized naming**: Parameter-encoded directories replace timestamps
- **Consistent files**: All analyses generate `results.json`, `summary.txt`, `dashboard.png`
- **Metadata tracking**: Complete run history with automatic incrementing
- **Legacy support**: `--legacy` flag maintains old timestamped format

### **Current System Features**
- **6 analysis programs**: All tested and validated
- **Standardized outputs**: Consistent structure across all analyses
- **Run management**: Automatic incrementing with metadata tracking
- **Performance metrics**: Comprehensive R² and RMSE reporting

This system provides comprehensive reimbursement data analysis with standardized outputs and flexible execution options. 