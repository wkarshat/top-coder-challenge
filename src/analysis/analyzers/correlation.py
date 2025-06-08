"""
Correlation Analyzer

Analyzes correlations between variables in the dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy.stats import pearsonr

from core.interfaces import IAnalyzer
from core.utils import calculate_correlation_matrix


class CorrelationAnalyzer(IAnalyzer):
    """Analyzer for correlation analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize correlation analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.method = config.get('method', 'pearson')
        self.focus_pairs = config.get('focus_pairs', [])
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform correlation analysis.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with correlation results
        """
        results = {}
        
        # Overall correlation matrix
        corr_matrix = calculate_correlation_matrix(data, self.method)
        if not corr_matrix.empty:
            results['correlation_matrix'] = corr_matrix.to_dict()
        
        # Focus pair correlations with p-values
        if self.focus_pairs:
            focus_results = {}
            for pair in self.focus_pairs:
                if len(pair) == 2 and all(col in data.columns for col in pair):
                    col1, col2 = pair
                    if (pd.api.types.is_numeric_dtype(data[col1]) and 
                        pd.api.types.is_numeric_dtype(data[col2])):
                        
                        # Remove NaN values for correlation calculation
                        clean_data = data[[col1, col2]].dropna()
                        if len(clean_data) > 2:
                            corr, p_value = pearsonr(clean_data[col1], 
                                                   clean_data[col2])
                            focus_results[f"{col1}_vs_{col2}"] = {
                                'correlation': float(corr),
                                'p_value': float(p_value),
                                'sample_size': len(clean_data)
                            }
            
            results['focus_pairs'] = focus_results
        
        # Summary statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            # Find strongest correlations
            if not corr_matrix.empty:
                # Get upper triangle (excluding diagonal)
                mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
                upper_tri = corr_matrix.where(mask)
                
                # Find max and min correlations
                max_corr = upper_tri.max().max()
                min_corr = upper_tri.min().min()
                
                # Find the pairs with max/min correlations
                max_pair = None
                min_pair = None
                
                for col in upper_tri.columns:
                    for idx in upper_tri.index:
                        if pd.notna(upper_tri.loc[idx, col]):
                            if upper_tri.loc[idx, col] == max_corr:
                                max_pair = (idx, col)
                            if upper_tri.loc[idx, col] == min_corr:
                                min_pair = (idx, col)
                
                results['summary'] = {
                    'strongest_positive': {
                        'pair': max_pair,
                        'correlation': float(max_corr)
                    } if max_pair else None,
                    'strongest_negative': {
                        'pair': min_pair,
                        'correlation': float(min_corr)
                    } if min_pair else None,
                    'total_pairs': int(len(numeric_cols) * (len(numeric_cols) - 1) / 2)
                }
        
        return results 