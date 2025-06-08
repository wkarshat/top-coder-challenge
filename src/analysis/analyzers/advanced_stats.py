"""
Advanced Statistical Analysis

Performs sophisticated statistical analysis including normality tests,
distribution analysis, hypothesis testing, and outlier detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy import stats
from scipy.stats import (
    shapiro, normaltest, kstest, jarque_bera,
    ttest_ind, mannwhitneyu, chi2_contingency,
    spearmanr, kendalltau
)

from core.interfaces import IAnalyzer
from core.utils import detect_outliers


class AdvancedStatisticalAnalyzer(IAnalyzer):
    """Advanced statistical analysis with hypothesis testing."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize advanced statistical analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.significance_level = config.get('significance_level', 0.05)
        self.normality_tests = config.get('normality_tests', ['shapiro', 'normaltest'])
        self.outlier_methods = config.get('outlier_methods', ['iqr', 'zscore'])
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform advanced statistical analysis.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with advanced statistical results
        """
        results = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            return {'error': 'No numeric columns found for analysis'}
        
        # Normality testing
        results['normality_tests'] = self._test_normality(data, numeric_cols)
        
        # Distribution analysis
        results['distribution_analysis'] = self._analyze_distributions(data, numeric_cols)
        
        # Outlier analysis
        results['outlier_analysis'] = self._analyze_outliers(data, numeric_cols)
        
        # Correlation alternatives
        results['correlation_alternatives'] = self._alternative_correlations(data, numeric_cols)
        
        # Data quality assessment
        results['data_quality'] = self._assess_data_quality(data)
        
        return results
    
    def _test_normality(self, data: pd.DataFrame, 
                       numeric_cols: List[str]) -> Dict[str, Any]:
        """Test normality of numeric columns."""
        normality_results = {}
        
        for col in numeric_cols:
            series = data[col].dropna()
            
            if len(series) < 3:
                normality_results[col] = {'error': 'Insufficient data'}
                continue
            
            col_results = {}
            
            # Shapiro-Wilk test (best for small samples)
            if 'shapiro' in self.normality_tests and len(series) <= 5000:
                try:
                    stat, p_value = shapiro(series)
                    col_results['shapiro'] = {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'is_normal': p_value > self.significance_level
                    }
                except Exception as e:
                    col_results['shapiro'] = {'error': str(e)}
            
            # D'Agostino's normality test
            if 'normaltest' in self.normality_tests and len(series) >= 8:
                try:
                    stat, p_value = normaltest(series)
                    col_results['dagostino'] = {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'is_normal': p_value > self.significance_level
                    }
                except Exception as e:
                    col_results['dagostino'] = {'error': str(e)}
            
            # Summary
            normal_tests = [test for test in col_results.values() 
                          if isinstance(test, dict) and 'is_normal' in test]
            if normal_tests:
                normal_count = sum(1 for test in normal_tests if test['is_normal'])
                col_results['summary'] = {
                    'likely_normal': normal_count > len(normal_tests) / 2,
                    'tests_passed': normal_count,
                    'total_tests': len(normal_tests)
                }
            
            normality_results[col] = col_results
        
        return normality_results
    
    def _analyze_distributions(self, data: pd.DataFrame, 
                             numeric_cols: List[str]) -> Dict[str, Any]:
        """Analyze distribution characteristics."""
        distribution_results = {}
        
        for col in numeric_cols:
            series = data[col].dropna()
            
            if len(series) < 3:
                distribution_results[col] = {'error': 'Insufficient data'}
                continue
            
            # Basic distribution statistics
            col_results = {
                'skewness': float(stats.skew(series)),
                'kurtosis': float(stats.kurtosis(series)),
                'variance': float(series.var()),
                'coefficient_of_variation': float(series.std() / series.mean()) if series.mean() != 0 else float('inf')
            }
            
            # Classify distribution shape
            skew = col_results['skewness']
            kurt = col_results['kurtosis']
            
            if abs(skew) < 0.5:
                skew_desc = 'approximately symmetric'
            elif skew > 0.5:
                skew_desc = 'right-skewed (positive)'
            else:
                skew_desc = 'left-skewed (negative)'
            
            if abs(kurt) < 0.5:
                kurt_desc = 'mesokurtic (normal-like)'
            elif kurt > 0.5:
                kurt_desc = 'leptokurtic (heavy-tailed)'
            else:
                kurt_desc = 'platykurtic (light-tailed)'
            
            col_results['distribution_shape'] = {
                'skewness_description': skew_desc,
                'kurtosis_description': kurt_desc
            }
            
            # Percentile analysis
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            col_results['percentiles'] = {
                f'p{p}': float(series.quantile(p/100)) for p in percentiles
            }
            
            distribution_results[col] = col_results
        
        return distribution_results
    
    def _analyze_outliers(self, data: pd.DataFrame, 
                         numeric_cols: List[str]) -> Dict[str, Any]:
        """Comprehensive outlier analysis."""
        outlier_results = {}
        
        for method in self.outlier_methods:
            try:
                outliers = detect_outliers(data[numeric_cols], method=method)
                
                method_results = {}
                for col in numeric_cols:
                    if col in outliers.columns:
                        outlier_mask = outliers[col]
                        outlier_count = outlier_mask.sum()
                        outlier_percentage = (outlier_count / len(data)) * 100
                        
                        method_results[col] = {
                            'outlier_count': int(outlier_count),
                            'outlier_percentage': float(outlier_percentage)
                        }
                
                outlier_results[method] = method_results
                
            except Exception as e:
                outlier_results[method] = {'error': str(e)}
        
        return outlier_results
    
    def _alternative_correlations(self, data: pd.DataFrame, 
                                numeric_cols: List[str]) -> Dict[str, Any]:
        """Calculate alternative correlation measures."""
        correlation_results = {}
        
        if len(numeric_cols) < 2:
            return {'error': 'Need at least 2 numeric columns'}
        
        # Spearman rank correlation
        try:
            spearman_matrix = data[numeric_cols].corr(method='spearman')
            correlation_results['spearman'] = spearman_matrix.to_dict()
        except Exception as e:
            correlation_results['spearman'] = {'error': str(e)}
        
        # Kendall tau correlation
        try:
            kendall_matrix = data[numeric_cols].corr(method='kendall')
            correlation_results['kendall'] = kendall_matrix.to_dict()
        except Exception as e:
            correlation_results['kendall'] = {'error': str(e)}
        
        return correlation_results
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality."""
        quality_results = {}
        
        # Missing data analysis
        missing_data = data.isnull().sum()
        total_cells = len(data) * len(data.columns)
        total_missing = missing_data.sum()
        
        quality_results['missing_data'] = {
            'total_missing_cells': int(total_missing),
            'missing_percentage': float((total_missing / total_cells) * 100),
            'complete_rows': int(len(data.dropna())),
            'complete_row_percentage': float((len(data.dropna()) / len(data)) * 100)
        }
        
        # Duplicate analysis
        quality_results['duplicates'] = {
            'duplicate_rows': int(data.duplicated().sum()),
            'duplicate_percentage': float((data.duplicated().sum() / len(data)) * 100)
        }
        
        return quality_results 