"""
Domain-agnostic interfaces for data analysis system.

This module defines abstract base classes that can be used for any
tabular data analysis, without domain-specific naming or assumptions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


# Data Types and Enums
class ModelType(Enum):
    """Enumeration of supported model types."""
    LINEAR = "linear"
    TREE = "tree"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


class AnalysisType(Enum):
    """Enumeration of analysis types."""
    CORRELATION = "correlation"
    STATISTICAL = "statistical"
    EXPLORATORY = "exploratory"


class OutputFormat(Enum):
    """Enumeration of supported output formats."""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PDF = "pdf"
    PNG = "png"
    SVG = "svg"


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class AnalysisResult:
    """Standard result structure for analyses."""
    analysis_type: AnalysisType
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelResult:
    """Result structure for model operations."""
    model_type: ModelType
    predictions: np.ndarray
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


# Core Interfaces

class IDataLoader(ABC):
    """Interface for data loading components."""
    
    @abstractmethod
    def load_and_process(self, source: str, config: dict) -> pd.DataFrame:
        """Load and process data from specified source."""
        pass


class IValidator(ABC):
    """Interface for data validation components."""
    
    @abstractmethod
    def validate(self, data: pd.DataFrame, 
                config: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate data according to configuration."""
        pass
    
    @abstractmethod
    def validate_schema(self, data: pd.DataFrame, 
                       required_columns: List[str]) -> ValidationResult:
        """Validate data schema."""
        pass


class IPreprocessor(ABC):
    """Interface for data preprocessing components."""
    
    @abstractmethod
    def process(self, data: pd.DataFrame, 
               config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Process data according to configuration."""
        pass
    
    @abstractmethod
    def handle_missing(self, data: pd.DataFrame, 
                      strategy: str = 'drop') -> pd.DataFrame:
        """Handle missing values."""
        pass
    
    @abstractmethod
    def detect_outliers(self, data: pd.DataFrame, 
                       method: str = 'iqr') -> pd.DataFrame:
        """Detect outliers in data."""
        pass


class IModel(ABC):
    """Base interface for all models."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def get_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate model metrics."""
        pass


class IAnalyzer(ABC):
    """Interface for analysis components."""
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        """Perform analysis on data."""
        pass


class IVisualizer(ABC):
    """Interface for visualization components."""
    
    @abstractmethod
    def create_plots(self, data: pd.DataFrame, analysis_results: dict,
                    model_results: dict) -> List[str]:
        """Create plots and return file paths."""
        pass


class IReporter(ABC):
    """Interface for report generation components."""
    
    @abstractmethod
    def generate_report(self, data, analysis_results: dict,
                       model_results: dict) -> str:
        """Generate analysis report."""
        pass


# Factory Interfaces

class IModelFactory(ABC):
    """Factory interface for creating model instances."""
    
    @abstractmethod
    def create_model(self, model_type: ModelType, 
                    config: Dict[str, Any]) -> IModel:
        """Create model instance."""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[ModelType]:
        """Get supported model types."""
        pass


class IAnalyzerFactory(ABC):
    """Factory interface for creating analyzer instances."""
    
    @abstractmethod
    def create_analyzer(self, analysis_type: AnalysisType, 
                       config: Dict[str, Any]) -> IAnalyzer:
        """Create analyzer instance."""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[AnalysisType]:
        """Get supported analysis types."""
        pass


# Exception Classes

class AnalysisError(Exception):
    """Base exception for analysis system."""
    pass


class DataValidationError(AnalysisError):
    """Exception for data validation errors."""
    pass


class ModelError(AnalysisError):
    """Exception for model-related errors."""
    pass


class ConfigurationError(AnalysisError):
    """Exception for configuration errors."""
    pass


class VisualizationError(AnalysisError):
    """Exception for visualization errors."""
    pass 