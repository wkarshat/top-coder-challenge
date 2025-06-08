"""
Time Series Analysis

Performs time series analysis including trend detection, seasonality analysis,
autocorrelation, and basic forecasting capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy import stats
from scipy.stats import linregress

from core.interfaces import IAnalyzer


class TimeSeriesAnalyzer(IAnalyzer):
    """Time series analysis for temporal data patterns."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize time series analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.time_column = config.get('time_column')
        self.value_columns = config.get('value_columns', [])
        self.window_size = config.get('window_size', 7)
        self.seasonal_periods = config.get('seasonal_periods', [7, 30, 365])
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform time series analysis.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with time series analysis results
        """
        results = {}
        
        # Identify time column if not specified
        if not self.time_column:
            time_cols = self._identify_time_columns(data)
            if time_cols:
                self.time_column = time_cols[0]
            else:
                return {'error': 'No time column found or specified'}
        
        if self.time_column not in data.columns:
            return {'error': f'Time column {self.time_column} not found'}
        
        # Prepare time series data
        ts_data = self._prepare_time_series(data)
        if ts_data is None:
            return {'error': 'Could not prepare time series data'}
        
        # Identify value columns if not specified
        if not self.value_columns:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            self.value_columns = [col for col in numeric_cols 
                                if col != self.time_column]
        
        if not self.value_columns:
            return {'error': 'No numeric value columns found'}
        
        # Trend analysis
        results['trend_analysis'] = self._analyze_trends(ts_data)
        
        # Seasonality analysis
        results['seasonality_analysis'] = self._analyze_seasonality(ts_data)
        
        # Autocorrelation analysis
        results['autocorrelation'] = self._analyze_autocorrelation(ts_data)
        
        # Moving averages
        results['moving_averages'] = self._calculate_moving_averages(ts_data)
        
        # Change point detection
        results['change_points'] = self._detect_change_points(ts_data)
        
        # Basic forecasting
        results['forecasting'] = self._basic_forecasting(ts_data)
        
        return results
    
    def _identify_time_columns(self, data: pd.DataFrame) -> List[str]:
        """Identify potential time columns."""
        time_columns = []
        
        for col in data.columns:
            # Check for datetime types
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                time_columns.append(col)
            # Check for common time column names
            elif any(keyword in col.lower() for keyword in 
                   ['time', 'date', 'day', 'month', 'year', 'period']):
                time_columns.append(col)
        
        return time_columns
    
    def _prepare_time_series(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare time series data with proper time index."""
        try:
            ts_data = data.copy()
            
            # Convert time column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(ts_data[self.time_column]):
                # Try to convert numeric time column (assuming it's days, etc.)
                if pd.api.types.is_numeric_dtype(ts_data[self.time_column]):
                    # Create a simple date range starting from a base date
                    base_date = pd.Timestamp('2020-01-01')
                    ts_data[self.time_column] = base_date + pd.to_timedelta(
                        ts_data[self.time_column], unit='D'
                    )
                else:
                    ts_data[self.time_column] = pd.to_datetime(
                        ts_data[self.time_column], errors='coerce'
                    )
            
            # Remove rows with invalid dates
            ts_data = ts_data.dropna(subset=[self.time_column])
            
            # Sort by time
            ts_data = ts_data.sort_values(self.time_column)
            
            # Set time as index
            ts_data.set_index(self.time_column, inplace=True)
            
            return ts_data
            
        except Exception:
            return None
    
    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in time series data."""
        trend_results = {}
        
        for col in self.value_columns:
            if col not in data.columns:
                continue
                
            series = data[col].dropna()
            if len(series) < 3:
                trend_results[col] = {'error': 'Insufficient data'}
                continue
            
            # Create numeric time index for regression
            time_numeric = np.arange(len(series))
            
            try:
                # Linear trend analysis
                slope, intercept, r_value, p_value, std_err = linregress(
                    time_numeric, series.values
                )
                
                # Trend classification
                if abs(slope) < std_err:
                    trend_type = 'stable'
                elif slope > 0:
                    trend_type = 'increasing'
                else:
                    trend_type = 'decreasing'
                
                # Calculate trend strength
                trend_strength = abs(r_value)
                
                # Detect trend changes
                window = min(len(series) // 4, 10)
                if window >= 3:
                    rolling_slopes = []
                    for i in range(window, len(series) - window):
                        subset_time = time_numeric[i-window:i+window]
                        subset_values = series.iloc[i-window:i+window].values
                        if len(subset_values) >= 3:
                            subset_slope, _, _, _, _ = linregress(
                                subset_time, subset_values
                            )
                            rolling_slopes.append(subset_slope)
                    
                    trend_volatility = np.std(rolling_slopes) if rolling_slopes else 0
                else:
                    trend_volatility = 0
                
                trend_results[col] = {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'trend_type': trend_type,
                    'trend_strength': float(trend_strength),
                    'trend_volatility': float(trend_volatility),
                    'is_significant': p_value < 0.05
                }
                
            except Exception as e:
                trend_results[col] = {'error': str(e)}
        
        return trend_results
    
    def _analyze_seasonality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonality patterns."""
        seasonality_results = {}
        
        for col in self.value_columns:
            if col not in data.columns:
                continue
                
            series = data[col].dropna()
            if len(series) < 10:
                seasonality_results[col] = {'error': 'Insufficient data'}
                continue
            
            col_results = {}
            
            # Test different seasonal periods
            for period in self.seasonal_periods:
                if len(series) >= period * 2:
                    try:
                        # Simple seasonal decomposition
                        seasonal_means = []
                        for i in range(period):
                            seasonal_subset = series.iloc[i::period]
                            if len(seasonal_subset) > 0:
                                seasonal_means.append(seasonal_subset.mean())
                        
                        if len(seasonal_means) == period:
                            # Calculate seasonal strength
                            overall_mean = series.mean()
                            seasonal_variance = np.var(seasonal_means)
                            total_variance = series.var()
                            
                            seasonal_strength = seasonal_variance / total_variance if total_variance > 0 else 0
                            
                            col_results[f'period_{period}'] = {
                                'seasonal_means': [float(x) for x in seasonal_means],
                                'seasonal_strength': float(seasonal_strength),
                                'is_seasonal': seasonal_strength > 0.1
                            }
                            
                    except Exception as e:
                        col_results[f'period_{period}'] = {'error': str(e)}
            
            seasonality_results[col] = col_results
        
        return seasonality_results
    
    def _analyze_autocorrelation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze autocorrelation patterns."""
        autocorr_results = {}
        
        for col in self.value_columns:
            if col not in data.columns:
                continue
                
            series = data[col].dropna()
            if len(series) < 10:
                autocorr_results[col] = {'error': 'Insufficient data'}
                continue
            
            try:
                # Calculate autocorrelations for different lags
                max_lag = min(len(series) // 4, 20)
                autocorrelations = []
                
                for lag in range(1, max_lag + 1):
                    if len(series) > lag:
                        corr = series.autocorr(lag=lag)
                        if not np.isnan(corr):
                            autocorrelations.append({
                                'lag': lag,
                                'correlation': float(corr)
                            })
                
                # Find significant autocorrelations
                significant_lags = [
                    ac for ac in autocorrelations 
                    if abs(ac['correlation']) > 0.2
                ]
                
                autocorr_results[col] = {
                    'autocorrelations': autocorrelations,
                    'significant_lags': significant_lags,
                    'max_autocorr': max(autocorrelations, 
                                      key=lambda x: abs(x['correlation'])) if autocorrelations else None
                }
                
            except Exception as e:
                autocorr_results[col] = {'error': str(e)}
        
        return autocorr_results
    
    def _calculate_moving_averages(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate moving averages."""
        ma_results = {}
        
        windows = [3, 7, 14, 30] if not hasattr(self, 'ma_windows') else self.ma_windows
        
        for col in self.value_columns:
            if col not in data.columns:
                continue
                
            series = data[col].dropna()
            if len(series) < 3:
                ma_results[col] = {'error': 'Insufficient data'}
                continue
            
            col_results = {}
            
            for window in windows:
                if len(series) >= window:
                    try:
                        ma = series.rolling(window=window).mean()
                        
                        # Calculate smoothness (how much MA reduces volatility)
                        original_std = series.std()
                        ma_std = ma.dropna().std()
                        smoothness = 1 - (ma_std / original_std) if original_std > 0 else 0
                        
                        col_results[f'ma_{window}'] = {
                            'values': ma.dropna().tolist()[-10:],  # Last 10 values
                            'smoothness': float(smoothness),
                            'current_value': float(ma.iloc[-1]) if not ma.empty else None
                        }
                        
                    except Exception as e:
                        col_results[f'ma_{window}'] = {'error': str(e)}
            
            ma_results[col] = col_results
        
        return ma_results
    
    def _detect_change_points(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect change points in time series."""
        change_point_results = {}
        
        for col in self.value_columns:
            if col not in data.columns:
                continue
                
            series = data[col].dropna()
            if len(series) < 10:
                change_point_results[col] = {'error': 'Insufficient data'}
                continue
            
            try:
                # Simple change point detection using rolling statistics
                window = max(len(series) // 10, 3)
                
                rolling_mean = series.rolling(window=window).mean()
                rolling_std = series.rolling(window=window).std()
                
                # Detect points where statistics change significantly
                mean_changes = []
                std_changes = []
                
                for i in range(window, len(series) - window):
                    before_mean = rolling_mean.iloc[i-1]
                    after_mean = rolling_mean.iloc[i+1]
                    before_std = rolling_std.iloc[i-1]
                    after_std = rolling_std.iloc[i+1]
                    
                    if (not np.isnan(before_mean) and not np.isnan(after_mean) and
                        not np.isnan(before_std) and not np.isnan(after_std)):
                        
                        mean_change = abs(after_mean - before_mean) / before_mean if before_mean != 0 else 0
                        std_change = abs(after_std - before_std) / before_std if before_std != 0 else 0
                        
                        if mean_change > 0.2:  # 20% change threshold
                            mean_changes.append({
                                'index': i,
                                'change_magnitude': float(mean_change)
                            })
                        
                        if std_change > 0.5:  # 50% change threshold
                            std_changes.append({
                                'index': i,
                                'change_magnitude': float(std_change)
                            })
                
                change_point_results[col] = {
                    'mean_change_points': mean_changes,
                    'volatility_change_points': std_changes,
                    'total_change_points': len(mean_changes) + len(std_changes)
                }
                
            except Exception as e:
                change_point_results[col] = {'error': str(e)}
        
        return change_point_results
    
    def _basic_forecasting(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform basic forecasting."""
        forecast_results = {}
        
        forecast_periods = self.config.get('forecast_periods', 5)
        
        for col in self.value_columns:
            if col not in data.columns:
                continue
                
            series = data[col].dropna()
            if len(series) < 5:
                forecast_results[col] = {'error': 'Insufficient data'}
                continue
            
            try:
                # Simple linear extrapolation
                time_numeric = np.arange(len(series))
                slope, intercept, r_value, _, _ = linregress(
                    time_numeric, series.values
                )
                
                # Generate forecasts
                future_time = np.arange(len(series), len(series) + forecast_periods)
                linear_forecast = slope * future_time + intercept
                
                # Moving average forecast
                ma_window = min(len(series), 5)
                recent_avg = series.tail(ma_window).mean()
                ma_forecast = [recent_avg] * forecast_periods
                
                # Trend-adjusted moving average
                recent_trend = slope
                trend_adjusted_forecast = [
                    recent_avg + recent_trend * (i + 1) 
                    for i in range(forecast_periods)
                ]
                
                forecast_results[col] = {
                    'linear_forecast': [float(x) for x in linear_forecast],
                    'moving_average_forecast': [float(x) for x in ma_forecast],
                    'trend_adjusted_forecast': [float(x) for x in trend_adjusted_forecast],
                    'forecast_confidence': float(r_value ** 2),
                    'last_actual_value': float(series.iloc[-1])
                }
                
            except Exception as e:
                forecast_results[col] = {'error': str(e)}
        
        return forecast_results 