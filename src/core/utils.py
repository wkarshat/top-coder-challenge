"""
Core Utilities

Generic utility functions for configuration, logging, data processing,
and system operations used throughout the analysis system.
"""

import yaml
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8', newline='\n') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif config_file.suffix.lower() == '.json':
                return json.load(f) or {}
            else:
                raise ValueError(f"Unsupported config format: {config_file.suffix}")
    
    except Exception as e:
        raise RuntimeError(f"Failed to load config {config_path}: {e}")


def create_timestamped_output_dir(base_dir: str = 'outputs', 
                                 prefix: str = 'analysis') -> str:
    """
    Create a timestamped output directory for organizing analysis results.
    
    Args:
        base_dir: Base output directory (default: 'outputs')
        prefix: Prefix for the timestamped directory (default: 'analysis')
        
    Returns:
        Path to the created timestamped directory
    """
    timestamp = datetime.now().strftime('%H%M%S')
    
    # Ensure base_dir is relative to project root, not src
    if not Path(base_dir).is_absolute():
        # If we're in src directory, go up one level
        current_dir = Path.cwd()
        if current_dir.name == 'src':
            base_dir = f"../{base_dir}"
    
    timestamped_dir = f"{base_dir}/{prefix}_{timestamp}"
    
    # Create the timestamped directory (no subdirectories)
    ensure_directories([timestamped_dir])
    
    return timestamped_dir


def setup_logging(logging_config: Dict[str, Any]) -> None:
    """
    Set up logging configuration.
    
    Args:
        logging_config: Logging configuration dictionary
    """
    level = logging_config.get('level', 'INFO').upper()
    log_file = logging_config.get('file')
    format_str = logging_config.get(
        'format', 
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level, logging.INFO)
    
    # Configure logging
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(format_str))
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        ensure_directories([str(Path(log_file).parent)])
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(format_str))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True  # Override any existing configuration
    )


def ensure_directories(paths: List[str]) -> None:
    """
    Ensure that directories exist, creating them if necessary.
    
    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if all required columns are present
    """
    missing_columns = set(required_columns) - set(df.columns)
    return len(missing_columns) == 0


def clean_numeric_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Clean numeric data by converting to numeric types and handling errors.
    
    Args:
        df: DataFrame to clean
        columns: List of column names to clean
        
    Returns:
        DataFrame with cleaned numeric columns
    """
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            # Convert to numeric, coercing errors to NaN
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean


def detect_outliers(df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
    """
    Detect outliers in numeric data using specified method.
    
    Args:
        df: DataFrame with numeric data
        method: Outlier detection method ('iqr', 'zscore', 'modified_zscore')
        
    Returns:
        DataFrame with boolean values indicating outliers
    """
    outliers = pd.DataFrame(False, index=df.index, columns=df.columns)
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers[col] = z_scores > 3
        
        elif method == 'modified_zscore':
            median = df[col].median()
            mad = np.median(np.abs(df[col] - median))
            modified_z_scores = 0.6745 * (df[col] - median) / mad
            outliers[col] = np.abs(modified_z_scores) > 3.5
    
    return outliers


def parallel_execute(tasks: List[Callable], max_workers: int = 4) -> List[Any]:
    """
    Execute tasks in parallel using ThreadPoolExecutor.
    
    Args:
        tasks: List of callable tasks to execute
        max_workers: Maximum number of worker threads
        
    Returns:
        List of results in the same order as tasks
    """
    if not tasks:
        return []
    
    if len(tasks) == 1 or max_workers == 1:
        # Execute sequentially if only one task or one worker
        return [task() for task in tasks]
    
    results = [None] * len(tasks)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks with their indices
        future_to_index = {
            executor.submit(task): i 
            for i, task in enumerate(tasks)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                results[index] = {'error': str(e)}
    
    return results


def calculate_correlation_matrix(df: pd.DataFrame, 
                                method: str = 'pearson') -> pd.DataFrame:
    """
    Calculate correlation matrix for numeric columns.
    
    Args:
        df: DataFrame with numeric data
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Correlation matrix DataFrame
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    
    return df[numeric_cols].corr(method=method)


def get_summary_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Get summary statistics for numeric columns.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with statistics for each numeric column
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = {}
    
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 0:
            stats[col] = {
                'count': len(series),
                'mean': float(series.mean()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'median': float(series.median()),
                'q25': float(series.quantile(0.25)),
                'q75': float(series.quantile(0.75))
            }
    
    return stats


def create_bins(series: pd.Series, n_bins: int = 10, 
               method: str = 'equal_width') -> pd.Series:
    """
    Create bins for a numeric series.
    
    Args:
        series: Numeric series to bin
        n_bins: Number of bins to create
        method: Binning method ('equal_width', 'equal_freq')
        
    Returns:
        Series with bin labels
    """
    if method == 'equal_width':
        return pd.cut(series, bins=n_bins, include_lowest=True)
    elif method == 'equal_freq':
        return pd.qcut(series, q=n_bins, duplicates='drop')
    else:
        raise ValueError(f"Unknown binning method: {method}")


def safe_divide(numerator: pd.Series, denominator: pd.Series, 
               fill_value: float = 0.0) -> pd.Series:
    """
    Safely divide two series, handling division by zero.
    
    Args:
        numerator: Numerator series
        denominator: Denominator series
        fill_value: Value to use when denominator is zero
        
    Returns:
        Series with division results
    """
    result = numerator / denominator
    result = result.fillna(fill_value)
    result = result.replace([np.inf, -np.inf], fill_value)
    return result


def format_number(value: float, precision: int = 3) -> str:
    """
    Format a number for display with appropriate precision.
    
    Args:
        value: Number to format
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    if pd.isna(value):
        return 'N/A'
    
    if abs(value) >= 1000:
        return f"{value:,.{precision}f}"
    else:
        return f"{value:.{precision}f}"


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save analysis results to file.
    
    Args:
        results: Results dictionary to save
        output_path: Path to output file
    """
    output_file = Path(output_path)
    ensure_directories([str(output_file.parent)])
    
    if output_file.suffix.lower() == '.json':
        with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(results, f, indent=2, default=str)
    elif output_file.suffix.lower() in ['.yaml', '.yml']:
        with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
            yaml.dump(results, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported output format: {output_file.suffix}")


def load_results(input_path: str) -> Dict[str, Any]:
    """
    Load analysis results from file.
    
    Args:
        input_path: Path to input file
        
    Returns:
        Results dictionary
    """
    input_file = Path(input_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")
    
    with open(input_file, 'r', encoding='utf-8', newline='\n') as f:
        if input_file.suffix.lower() == '.json':
            return json.load(f)
        elif input_file.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f) or {}
        else:
            raise ValueError(f"Unsupported input format: {input_file.suffix}")


def create_series_output_dir(base_name: str, series_id: str = None, base_dir: str = "outputs") -> Path:
    """
    Create a unique output directory for a series of analysis runs.
    
    Args:
        base_name: Base name for the analysis (e.g., 'comprehensive_analysis')
        series_id: Optional series identifier (defaults to current date)
        base_dir: Base output directory
        
    Returns:
        Path to the created series directory
        
    Example:
        create_series_output_dir('daily_analysis') 
        -> outputs/daily_analysis_20250608/run_001/
    """
    if series_id is None:
        series_id = datetime.now().strftime("%Y%m%d")
    
    series_dir = Path(base_dir) / f"{base_name}_{series_id}"
    series_dir.mkdir(parents=True, exist_ok=True)
    
    # Find next run number
    existing_runs = [d for d in series_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    if existing_runs:
        run_numbers = [int(d.name.split('_')[1]) for d in existing_runs if d.name.split('_')[1].isdigit()]
        next_run = max(run_numbers) + 1 if run_numbers else 1
    else:
        next_run = 1
    
    run_dir = series_dir / f"run_{next_run:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create series metadata
    metadata_file = series_dir / "series_metadata.json"
    if not metadata_file.exists():
        metadata = {
            "series_name": base_name,
            "series_id": series_id,
            "created": datetime.now().isoformat(),
            "runs": []
        }
    else:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    # Add current run to metadata
    metadata["runs"].append({
        "run_number": next_run,
        "run_dir": str(run_dir),
        "timestamp": datetime.now().isoformat()
    })
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return run_dir


def create_timestamped_output_dir(analysis_type: str = "analysis", base_dir: str = "outputs") -> Path:
    """
    Create a timestamped output directory (legacy format for compatibility).
    
    Args:
        analysis_type: Type of analysis
        base_dir: Base output directory
        
    Returns:
        Path to the created timestamped directory
    """
    timestamp = datetime.now().strftime("%H%M%S")
    output_dir = Path(base_dir) / f"{analysis_type}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def get_latest_series_run(base_name: str, series_id: str = None, base_dir: str = "outputs") -> Path:
    """
    Get the latest run directory for a given series.
    
    Args:
        base_name: Base name for the analysis
        series_id: Series identifier (defaults to current date)
        base_dir: Base output directory
        
    Returns:
        Path to the latest run directory, or None if no runs exist
    """
    if series_id is None:
        series_id = datetime.now().strftime("%Y%m%d")
    
    series_dir = Path(base_dir) / f"{base_name}_{series_id}"
    if not series_dir.exists():
        return None
    
    existing_runs = [d for d in series_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    if not existing_runs:
        return None
    
    run_numbers = [(int(d.name.split('_')[1]), d) for d in existing_runs if d.name.split('_')[1].isdigit()]
    if not run_numbers:
        return None
    
    latest_run = max(run_numbers, key=lambda x: x[0])[1]
    return latest_run


def list_series_runs(base_name: str, series_id: str = None, base_dir: str = "outputs") -> list:
    """
    List all runs in a series.
    
    Args:
        base_name: Base name for the analysis
        series_id: Series identifier (defaults to current date)
        base_dir: Base output directory
        
    Returns:
        List of run directories sorted by run number
    """
    if series_id is None:
        series_id = datetime.now().strftime("%Y%m%d")
    
    series_dir = Path(base_dir) / f"{base_name}_{series_id}"
    if not series_dir.exists():
        return []
    
    existing_runs = [d for d in series_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    run_numbers = [(int(d.name.split('_')[1]), d) for d in existing_runs if d.name.split('_')[1].isdigit()]
    
    return [run_dir for _, run_dir in sorted(run_numbers)] 