"""
Unified Data Loader with Validation and Preprocessing

Consolidates data loading, validation, and preprocessing into a single
component to reduce complexity while maintaining flexibility.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import logging

from core.interfaces import IDataLoader
from core.utils import validate_columns, clean_numeric_data, detect_outliers

logger = logging.getLogger(__name__)


class DataLoader(IDataLoader):
    """Unified data loader handling loading, validation, and preprocessing."""
    
    def __init__(self):
        self.supported_formats = {'.csv', '.json', '.xlsx', '.parquet'}
    
    def load_and_process(self, source: str, config: dict) -> pd.DataFrame:
        """
        Load data from source and apply validation and preprocessing.
        
        Args:
            source: Path to data file
            config: Configuration dictionary with data settings
            
        Returns:
            Processed DataFrame ready for analysis
        """
        logger.info(f"Loading data from {source}")
        
        # 1. Load raw data
        data = self._load_file(source)
        logger.info(f"Loaded {len(data)} records with "
                    f"{len(data.columns)} columns")
        
        # 2. Validate data structure
        self._validate_data(data, config)
        logger.info("Data validation passed")
        
        # 3. Preprocess data
        data = self._preprocess_data(data, config)
        logger.info(f"Preprocessing complete: {len(data)} records remaining")
        
        return data
    
    def _load_file(self, source: str) -> pd.DataFrame:
        """Load data from file based on extension."""
        source_path = Path(source)
        
        # Try current directory first, then data/ subdirectory
        if not source_path.exists():
            data_path = Path('data') / source_path.name
            if data_path.exists():
                source_path = data_path
            else:
                raise FileNotFoundError(f"Data file not found: {source}")
        
        suffix = source_path.suffix.lower()
        
        if suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        try:
            if suffix == '.csv':
                return pd.read_csv(source_path)
            elif suffix == '.json':
                return pd.read_json(source_path)
            elif suffix == '.xlsx':
                return pd.read_excel(source_path)
            elif suffix == '.parquet':
                return pd.read_parquet(source_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load {source_path}: {e}")
    
    def _validate_data(self, data: pd.DataFrame, config: dict) -> None:
        """Validate data structure and content."""
        validation_config = config.get('validation', {})
        
        # Check required columns
        required_columns = config.get('required_columns', [])
        if required_columns and not validate_columns(data, 
                                                     required_columns):
            missing = set(required_columns) - set(data.columns)
            raise ValueError(f"Missing required columns: {missing}")
        
        # Check minimum data requirements
        min_records = validation_config.get('min_records', 1)
        if len(data) < min_records:
            raise ValueError(f"Insufficient data: {len(data)} < "
                           f"{min_records}")
        
        # Check for completely empty columns
        empty_cols = data.columns[data.isnull().all()].tolist()
        if empty_cols and validation_config.get('strict_mode', False):
            raise ValueError(f"Empty columns found: {empty_cols}")
        
        logger.debug("Data validation checks passed")
    
    def _preprocess_data(self, data: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Apply preprocessing steps to clean and prepare data."""
        preprocessing_config = config.get('preprocessing', {})
        data = data.copy()
        
        # Handle missing values
        missing_strategy = preprocessing_config.get('handle_missing', 'drop')
        if missing_strategy == 'drop':
            initial_count = len(data)
            data = data.dropna()
            dropped = initial_count - len(data)
            if dropped > 0:
                logger.info(f"Dropped {dropped} rows with missing values")
        elif missing_strategy == 'fill':
            fill_value = preprocessing_config.get('fill_value', 0)
            data = data.fillna(fill_value)
            logger.info(f"Filled missing values with {fill_value}")
        
        # Clean numeric columns
        numeric_columns = self._identify_numeric_columns(data, config)
        if numeric_columns:
            data = clean_numeric_data(data, numeric_columns)
            logger.debug(f"Cleaned numeric columns: {numeric_columns}")
        
        # Handle outliers
        outlier_method = preprocessing_config.get('outlier_detection')
        if outlier_method and numeric_columns:
            outlier_data = detect_outliers(data[numeric_columns], method=outlier_method)
            outlier_action = preprocessing_config.get('outlier_action', 'flag')
            
            if outlier_action == 'remove':
                # Remove rows with any outliers
                outlier_mask = outlier_data.any(axis=1)
                initial_count = len(data)
                data = data[~outlier_mask]
                removed = initial_count - len(data)
                if removed > 0:
                    logger.info(f"Removed {removed} outlier records")
            elif outlier_action == 'flag':
                # Add outlier flags as new columns
                for col in numeric_columns:
                    if col in outlier_data.columns:
                        data[f'{col}_outlier'] = outlier_data[col]
                logger.debug("Added outlier flag columns")
        
        # Create derived features
        if preprocessing_config.get('create_ratios', False):
            data = self._create_ratio_features(data, numeric_columns)
        
        # Apply custom transformations
        transformations = preprocessing_config.get('transformations', [])
        for transform in transformations:
            data = self._apply_transformation(data, transform)
        
        return data
    
    def _identify_numeric_columns(self, data: pd.DataFrame, config: dict) -> List[str]:
        """Identify numeric columns for processing."""
        # Use explicitly specified columns if available
        feature_columns = config.get('feature_columns', [])
        if feature_columns:
            return [col for col in feature_columns if col in data.columns 
                   and pd.api.types.is_numeric_dtype(data[col])]
        
        # Otherwise, detect numeric columns
        return data.select_dtypes(include=[np.number]).columns.tolist()
    
    def _create_ratio_features(self, data: pd.DataFrame, 
                              numeric_columns: List[str]) -> pd.DataFrame:
        """Create ratio features between numeric columns."""
        data = data.copy()
        
        # Create ratios between pairs of columns
        for i, col1 in enumerate(numeric_columns):
            for col2 in numeric_columns[i+1:]:
                # Avoid division by zero
                mask = data[col2] != 0
                if mask.any():
                    ratio_name = f'{col1}_{col2}_ratio'
                    data.loc[mask, ratio_name] = data.loc[mask, col1] / data.loc[mask, col2]
                    logger.debug(f"Created ratio feature: {ratio_name}")
        
        return data
    
    def _apply_transformation(self, data: pd.DataFrame, 
                            transform_config: dict) -> pd.DataFrame:
        """Apply a single transformation to the data."""
        transform_type = transform_config.get('type')
        column = transform_config.get('column')
        
        if not column or column not in data.columns:
            logger.warning(f"Transformation skipped: column '{column}' not found")
            return data
        
        data = data.copy()
        
        if transform_type == 'log':
            # Log transformation (handle zeros/negatives)
            mask = data[column] > 0
            if mask.any():
                new_col = f'{column}_log'
                data.loc[mask, new_col] = np.log(data.loc[mask, column])
                logger.debug(f"Applied log transformation: {new_col}")
        
        elif transform_type == 'sqrt':
            # Square root transformation (handle negatives)
            mask = data[column] >= 0
            if mask.any():
                new_col = f'{column}_sqrt'
                data.loc[mask, new_col] = np.sqrt(data.loc[mask, column])
                logger.debug(f"Applied sqrt transformation: {new_col}")
        
        elif transform_type == 'normalize':
            # Min-max normalization
            col_min = data[column].min()
            col_max = data[column].max()
            if col_max > col_min:
                new_col = f'{column}_norm'
                data[new_col] = (data[column] - col_min) / (col_max - col_min)
                logger.debug(f"Applied normalization: {new_col}")
        
        elif transform_type == 'standardize':
            # Z-score standardization
            col_mean = data[column].mean()
            col_std = data[column].std()
            if col_std > 0:
                new_col = f'{column}_std'
                data[new_col] = (data[column] - col_mean) / col_std
                logger.debug(f"Applied standardization: {new_col}")
        
        return data


class MultiSourceLoader(DataLoader):
    """Extended loader that can handle multiple data sources."""
    
    def load_multiple_sources(self, sources: List[str], 
                            config: dict) -> pd.DataFrame:
        """Load and combine data from multiple sources."""
        dataframes = []
        
        for source in sources:
            try:
                df = self.load_and_process(source, config)
                df['_source'] = source  # Track data source
                dataframes.append(df)
                logger.info(f"Loaded {len(df)} records from {source}")
            except Exception as e:
                logger.error(f"Failed to load {source}: {e}")
                if config.get('validation', {}).get('strict_mode', False):
                    raise
        
        if not dataframes:
            raise RuntimeError("No data sources could be loaded")
        
        # Combine all dataframes
        combined = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Combined {len(dataframes)} sources: {len(combined)} total records")
        
        return combined 