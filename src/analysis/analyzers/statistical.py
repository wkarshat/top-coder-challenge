"""
Statistical Analyzer

Performs basic statistical analysis on the dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from core.interfaces import IAnalyzer
from core.utils import get_summary_statistics


class StatisticalAnalyzer(IAnalyzer):
    """Analyzer for basic statistical analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize statistical analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform statistical analysis.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with statistical results
        """
        results = {}
        
        # Basic dataset info
        results['dataset_info'] = {
            'total_records': len(data),
            'total_columns': len(data.columns),
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'missing_values': data.isnull().sum().sum()
        }
        
        # Summary statistics for numeric columns
        results['summary_statistics'] = get_summary_statistics(data)
        
        # Missing value analysis
        missing_analysis = {}
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                missing_analysis[col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_count / len(data) * 100)
                }
        results['missing_values'] = missing_analysis
        
        # Data type analysis
        dtype_analysis = {}
        for col in data.columns:
            dtype_analysis[col] = str(data[col].dtype)
        results['data_types'] = dtype_analysis
        
        return results 