"""
Linear Model Implementation

Simple linear regression model for testing the modular architecture.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Any

from core.interfaces import IModel


class LinearModel(IModel):
    """Linear regression model implementation."""
    
    def __init__(self, fit_intercept: bool = True, **kwargs):
        """
        Initialize linear model.
        
        Args:
            fit_intercept: Whether to fit intercept
            **kwargs: Additional parameters
        """
        self.fit_intercept = fit_intercept
        self.model = LinearRegression(fit_intercept=fit_intercept)
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the linear model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Self for method chaining
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def get_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate model metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        return {
            'r2_score': float(r2_score(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred)))
        }
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        result = {'intercept': float(self.model.intercept_)}
        if hasattr(self.model, 'coef_'):
            result['coefficients'] = self.model.coef_.tolist()
        
        return result 