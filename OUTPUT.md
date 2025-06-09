# Three Variable Fitting 809 Data - Output Management

This document describes the complete output management system for the Legacy Analysis System, including directory structure, file organization, logging, test results, and best practices.

## Overview

All analysis outputs are organized using **standardized series-based directories** under the `outputs/` directory. This ensures:
- **Parameter-encoded naming** for easy identification
- **Run-based organization** with automatic incrementing
- **Standardized file names** across all analyses
- **Metadata tracking** for complete run history
- **No file conflicts** between different analysis runs

## Directory Structure

### Current Standardized Structure

```
outputs/
├── README.md                           # Documentation
├── comprehensive_r8_20250608/          # 8-region comprehensive analysis
│   ├── metadata.json (325B)           # Run tracking metadata
│   ├── run001/                        # First run
│   │   ├── results.json (48KB)        # Standardized results
│   │   ├── summary.txt (151B)         # Human-readable summary
│   │   └── dashboard.png (411KB)      # Primary visualization
│   └── run002/                        # Second run
│       ├── results.json
│       ├── summary.txt
│       └── dashboard.png
├── daily_m14_20250608/                # Daily analysis (max 14 days)
│   ├── metadata.json (312B)
│   └── run001/
│       ├── results.json (68KB)
│       ├── summary.txt (143B)
│       └── dashboard.png (437KB)
├── daily_rate_m150_r100_20250608/     # Daily rate (150 miles, $100 receipts)
│   ├── metadata.json (348B)
│   └── run001/
│       ├── results.json (65KB)
│       ├── summary.txt (148B)
│       └── dashboard.png (544KB)
├── main_20250608/                     # Main system orchestrator
│   ├── metadata.json (282B)
│   └── run001/
│       ├── results.json (535KB)
│       ├── summary.txt (136B)
│       ├── correlation_heatmap.png (30KB)
│       ├── scatter_Miles_vs_Reimb.png (105KB)
│       └── scatter_Receipts_vs_Reimb.png (105KB)
├── reimb_threshold_t1000_20250608/    # $1000 threshold analysis
│   ├── metadata.json (337B)
│   └── run001/
│       ├── results.json (21KB)
│       ├── summary.txt (153B)
│       └── dashboard.png (383KB)
├── script_20250608/                   # Script wrapper
│   ├── metadata.json (286B)
│   └── run001/
│       ├── results.json (535KB)
│       ├── summary.txt (136B)
│       ├── correlation_heatmap.png (30KB)
│       ├── scatter_Miles_vs_Reimb.png (105KB)
│       └── scatter_Receipts_vs_Reimb.png (105KB)
└── logs/                              # System logs
    └── analysis.log (2.8KB)
```

**Total Files Generated**: 42 files across 6 analysis series  
**Total Data Size**: ~2.8MB of analysis results and visualizations

## Naming Conventions

### Directory Names
- **Format**: `{analysis_type}_{parameters}_{series_id}/`
- **Series ID**: `YYYYMMDD` (date-based identifier)
- **Parameter encoding**:
  - `r8` = 8 regions
  - `m14` = maximum 14 days
  - `m150_r100` = 150 miles threshold, $100 receipts threshold
  - `t1000` = $1000 threshold

### Run Directories
- **Format**: `run{number:03d}/` (run001, run002, etc.)
- **Auto-incrementing**: System automatically creates next available run number

### File Names (Standardized)
- **`results.json`** - Complete analysis results and metadata
- **`summary.txt`** - Human-readable analysis summary
- **`dashboard.png`** - Primary visualization dashboard
- **Additional files** - Analysis-specific outputs (correlation plots, etc.)

## Analysis Types and Commands

### 1. Main System Analysis (`main_20250608/`)

**Command**: `python src/core/main.py --csv public.csv --config config.yaml --analysis-config analysis.yaml`

**Outputs**:
- `results.json` (535KB) - Complete analysis results from 5 analyzers and 2 models
- `summary.txt` (136B) - Analysis summary with performance metrics
- `correlation_heatmap.png` (30KB) - Correlation matrix visualization
- `scatter_Miles_vs_Reimb.png` (105KB) - Miles vs Reimbursement scatter plot
- `scatter_Receipts_vs_Reimb.png` (105KB) - Receipts vs Reimbursement scatter plot

**Performance**: Linear R² = 0.784, Ensemble R² = 0.913  
**Test Status**: ✅ SUCCESS

### 2. Script Wrapper (`script_20250608/`)

**Command**: `python scripts/run_analysis.py --csv public.csv`

**Outputs**: Identical to main system (wrapper for same orchestrator)  
**Performance**: Linear R² = 0.784, Ensemble R² = 0.913  
**Test Status**: ✅ SUCCESS

### 3. Comprehensive Binned Analysis (`comprehensive_r8_20250608/`)

**Command**: `python -m src.analysis.binned.comprehensive_analysis --csv public.csv`

**Outputs**:
- `results.json` (48KB) - 8-region analysis with linear and polynomial models
- `summary.txt` (151B) - Regional performance summary
- `dashboard.png` (411KB) - Comprehensive visualization dashboard

**Performance**: Linear R² = 0.784, Polynomial avg R² = 0.852  
**Regions**: 8 regions analyzed (R0-R7)  
**Test Status**: ✅ SUCCESS

### 4. Daily Analysis (`daily_m14_20250608/`)

**Command**: `python -m src.analysis.binned.daily_analysis --csv public.csv`

**Outputs**:
- `results.json` (68KB) - Day-by-day analysis (Days 1-14)
- `summary.txt` (143B) - Daily performance ranking
- `dashboard.png` (437KB) - Daily analysis visualization

**Performance**: R² range 0.585-0.872 across 14 days  
**Best**: Day 2 (R² = 0.872), Day 3 (R² = 0.850)  
**Test Status**: ✅ SUCCESS

### 5. Daily Rate Analysis (`daily_rate_m150_r100_20250608/`)

**Command**: `python -m src.analysis.binned.daily_rate_analysis --csv public.csv`

**Outputs**:
- `results.json` (65KB) - 56 subregion analysis with rate thresholds
- `summary.txt` (148B) - Subregion performance summary
- `dashboard.png` (544KB) - Rate analysis visualization

**Performance**: 42/56 subregions populated, R² range 0.090-0.951  
**Best**: Day2_M≤300_R≤$200 (R² = 0.951)  
**Test Status**: ✅ SUCCESS

### 6. Reimbursement Threshold Analysis (`reimb_threshold_t1000_20250608/`)

**Command**: `python -m src.analysis.binned.reimb_threshold_analysis --csv public.csv`

**Outputs**:
- `results.json` (21KB) - Above/below $1000 threshold analysis
- `summary.txt` (153B) - Threshold comparison summary
- `dashboard.png` (383KB) - Threshold analysis visualization

**Performance**: Below $1000: R² = 0.612, Above $1000: R² = 0.619  
**Distribution**: 246 records below, 754 records above $1000 threshold  
**Test Status**: ✅ SUCCESS

## Test Results Summary

All 6 analysis programs successfully executed with standardized naming and proper directory organization:

| Analysis Program | Directory | Performance | Files | Status |
|-----------------|-----------|-------------|-------|--------|
| Main System | `main_20250608/` | Linear R² = 0.784, Ensemble R² = 0.913 | 5 files | ✅ SUCCESS |
| Script Wrapper | `script_20250608/` | Linear R² = 0.784, Ensemble R² = 0.913 | 5 files | ✅ SUCCESS |
| Comprehensive | `comprehensive_r8_20250608/` | Linear R² = 0.784, Polynomial avg R² = 0.852 | 3 files | ✅ SUCCESS |
| Daily Analysis | `daily_m14_20250608/` | R² range 0.585-0.872 | 3 files | ✅ SUCCESS |
| Daily Rate | `daily_rate_m150_r100_20250608/` | R² range 0.090-0.951 | 3 files | ✅ SUCCESS |
| Threshold | `reimb_threshold_t1000_20250608/` | Below: R² = 0.612, Above: R² = 0.619 | 3 files | ✅ SUCCESS |

## Metadata Tracking

Each analysis series includes `metadata.json` with complete run history:

```json
{
  "analysis_type": "comprehensive",
  "series_id": "20250608",
  "parameters": {"regions": 8},
  "created": "2025-06-08T20:33:26.911641",
  "runs": [
    {
      "run_number": 1,
      "run_dir": "outputs\\comprehensive_r8_20250608\\run001",
      "timestamp": "2025-06-08T20:33:26.911641"
    },
    {
      "run_number": 2,
      "run_dir": "outputs\\comprehensive_r8_20250608\\run002",
      "timestamp": "2025-06-08T20:36:26.252079"
    }
  ]
}
```

## File Contents

### `results.json`
Complete analysis results including:
- Model performance metrics (R², RMSE, coefficients)
- Statistical analysis results
- Regional/daily breakdowns
- Configuration parameters
- Data summary statistics

### `summary.txt`
Human-readable summary with:
- Analysis type and configuration
- Key performance metrics
- File generation confirmation
- Record counts and distributions

### `dashboard.png`
Primary visualization containing:
- Performance comparisons
- Regional/daily breakdowns
- Model coefficient visualizations
- Statistical distributions

## Legacy Format Support

The system supports legacy timestamped format using the `--legacy` flag:

**Command**: `python -m src.analysis.binned.comprehensive_analysis --csv public.csv --legacy`

**Creates**: `outputs/comprehensive_205818/` (legacy HHMMSS format)

## Configuration

### Automatic Directory Creation

```python
from core.utils import create_organized_output_dir

# Creates organized directory with parameter encoding
output_dir = create_organized_output_dir('comprehensive', {'regions': 8})
# Result: outputs/comprehensive_r8_20250608/run001/
```

### Standardized Results Saving

```python
from core.utils import save_standardized_results

# Save with standardized file names
save_standardized_results(output_dir, results_data, summary_text, dashboard_plot)
# Creates: results.json, summary.txt, dashboard.png
```

## Benefits

### ✅ **Parameter Identification**
- Directory names encode key parameters
- Easy to identify analysis configurations
- Quick comparison between parameter variations

### ✅ **Run Management**
- Automatic run numbering prevents conflicts
- Complete run history with metadata
- Easy to track analysis evolution

### ✅ **Standardization**
- Consistent file names across all analyses
- Predictable structure for automation
- Simplified result processing

### ✅ **Backward Compatibility**
- Legacy format still supported
- Gradual migration path
- No disruption to existing workflows

## Usage Examples

### Standard Analysis
```bash
python -m src.analysis.binned.comprehensive_analysis --csv public.csv
# Creates: outputs/comprehensive_r8_20250608/run001/
```

### Multiple Runs
```bash
python -m src.analysis.binned.comprehensive_analysis --csv public.csv
# Creates: outputs/comprehensive_r8_20250608/run002/
```

### Legacy Format
```bash
python -m src.analysis.binned.comprehensive_analysis --csv public.csv --legacy
# Creates: outputs/comprehensive_205818/
```

## Migration from Old Structure

**Before (Legacy)**:
```
outputs/
├── analysis_143208/              # Inconsistent naming
├── daily_analysis/               # No timestamp
├── reimb_threshold_analysis/     # No timestamp
└── analysis_data.json           # Non-standard file names
```

**After (Standardized)**:
```
outputs/
├── comprehensive_r8_20250608/
│   ├── metadata.json
│   └── run001/
│       ├── results.json         # Standardized names
│       ├── summary.txt
│       └── dashboard.png
├── daily_m14_20250608/
└── reimb_threshold_t1000_20250608/
```

## Logging System

### Log Configuration

Centralized logging in `outputs/logs/analysis.log`:

```yaml
logging:
  level: INFO
  format: '%(asctime)s - %(levelname)s - %(message)s'
  file: outputs/logs/analysis.log
  console: true
```

### Log Content

- **Directory creation**: Output path and run number
- **Analysis execution**: Start/completion times, performance metrics
- **File generation**: Results, visualizations, summaries
- **Error handling**: Detailed error messages and stack traces

### Sample Log Entries

```
2025-06-08 21:09:10,815 - INFO - Starting analysis pipeline for public.csv
2025-06-08 21:09:10,820 - INFO - Loaded 1000 records with 4 columns
2025-06-08 21:09:13,848 - INFO - linear model completed
2025-06-08 21:09:14,938 - INFO - ensemble model completed
2025-06-08 21:09:15,304 - INFO - Analysis pipeline completed successfully
```

## Best Practices

1. **Use standardized format** for new analyses (default behavior)
2. **Include parameter encoding** in directory names for identification
3. **Maintain run history** through metadata tracking
4. **Use consistent file names** (results.json, summary.txt, dashboard.png)
5. **Archive old series** periodically to save space
6. **Document significant findings** in summary.txt files
7. **Monitor log files** for analysis execution status
8. **Use PowerShell syntax** for Windows command execution

## Archive Policy

- **Keep recent series**: Last 5 series per analysis type
- **Archive older results**: Move to `archive/` subdirectory
- **Clean up logs**: Rotate analysis.log when it exceeds 10MB
- **Preserve metadata**: Always keep metadata.json files for historical tracking

## Program-to-Directory Mapping

| Directory Prefix | Source Program | Module Path | Analysis Type |
|-----------------|----------------|-------------|---------------|
| `comprehensive_` | comprehensive_analysis.py | src.analysis.binned.comprehensive_analysis | 8-region binned |
| `daily_m` | daily_analysis.py | src.analysis.binned.daily_analysis | Daily (1-14) |
| `daily_rate_` | daily_rate_analysis.py | src.analysis.binned.daily_rate_analysis | Subregion rate |
| `main_` | main.py | src.core.main | System orchestrator |
| `reimb_threshold_` | reimb_threshold_analysis.py | src.analysis.binned.reimb_threshold_analysis | Threshold split |
| `script_` | run_analysis.py | scripts.run_analysis | Script wrapper |

**Note**: `main_` and `script_` generate identical analysis results since `run_analysis.py` is a wrapper that calls the same `AnalysisOrchestrator` as `main.py`.

This standardized output management system provides consistent, organized, and maintainable output handling across all Legacy Analysis System components. 