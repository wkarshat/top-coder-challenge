"""
Domain-agnostic modular analysis script.

This script demonstrates how to build a flexible analysis pipeline
that can work with any tabular data, not just reimbursement data.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import os


@dataclass
class AnalysisConfig:
    """Configuration for analysis pipeline."""
    data_source: str
    target_column: str
    feature_columns: List[str]
    required_columns: List[str]
    analysis_types: List[str]
    model_types: List[str]
    output_dir: str = "outputs"
    enable_plots: bool = True


class DataLoader:
    """Generic data loading component."""
    
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        """Load data from various sources."""
        if source.endswith('.csv'):
            return self._load_csv(source, **kwargs)
        elif source.endswith('.json'):
            return self._load_json(source, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {source}")
    
    def _load_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load CSV file."""
        try:
            df = pd.read_csv(filepath, **kwargs)
            print(f"Loaded CSV: {filepath}, shape: {df.shape}")
            return df
        except FileNotFoundError:
            # Try data/ subdirectory
            data_path = os.path.join('data', os.path.basename(filepath))
            df = pd.read_csv(data_path, **kwargs)
            print(f"Loaded CSV: {data_path}, shape: {df.shape}")
            return df
    
    def _load_json(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load JSON file."""
        df = pd.read_json(filepath, **kwargs)
        print(f"Loaded JSON: {filepath}, shape: {df.shape}")
        return df


class DataValidator:
    """Generic data validation component."""
    
    def validate(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data according to configuration."""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required columns
        required_cols = config.get('required_columns', [])
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            results['errors'].append(f"Missing required columns: {missing_cols}")
            results['is_valid'] = False
        
        # Convert numeric columns
        numeric_cols = [col for col in required_cols if col in data.columns]
        for col in numeric_cols:
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            except Exception as e:
                results['warnings'].append(f"Could not convert {col}: {e}")
        
        return results


class LinearModel:
    """Linear regression model component."""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series, config: Dict[str, Any] = None):
        """Train the linear model."""
        from sklearn.linear_model import LinearRegression
        
        self.model = LinearRegression()
        self.model.fit(X, y)
        self.feature_names = list(X.columns)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def get_metrics(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate model metrics."""
        from sklearn.metrics import r2_score, mean_absolute_error
        
        predictions = self.predict(X)
        return {
            'r2_score': r2_score(y, predictions),
            'mae': mean_absolute_error(y, predictions)
        }


class CorrelationAnalyzer:
    """Correlation analysis component."""
    
    def analyze(self, data: pd.DataFrame, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform correlation analysis."""
        from scipy.stats import pearsonr
        
        results = {'pairwise_correlations': {}, 'significant_pairs': []}
        
        # Get focus pairs from config or analyze all numeric columns
        focus_pairs = config.get('focus_pairs', []) if config else []
        if not focus_pairs:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            focus_pairs = [(col1, col2) for i, col1 in enumerate(numeric_cols) 
                          for col2 in numeric_cols[i+1:]]
        
        for col1, col2 in focus_pairs:
            if col1 in data.columns and col2 in data.columns:
                corr, p_value = pearsonr(data[col1], data[col2])
                pair_key = f'{col1}_vs_{col2}'
                results['pairwise_correlations'][pair_key] = {
                    'correlation': corr,
                    'p_value': p_value
                }
                
                if p_value < 0.05 and abs(corr) > 0.3:
                    results['significant_pairs'].append({
                        'variables': (col1, col2),
                        'correlation': corr,
                        'p_value': p_value
                    })
        
        return results


class Visualizer:
    """Visualization component."""
    
    def __init__(self, output_dir: str = "outputs/plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_scatter_plot(self, data: pd.DataFrame, x_col: str, y_col: str, 
                           filename: str = None) -> str:
        """Create scatter plot with regression line."""
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        if filename is None:
            filename = f'{x_col}_vs_{y_col}.png'
        
        filepath = os.path.join(self.output_dir, filename)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        ax.scatter(data[x_col], data[y_col], alpha=0.6, s=30)
        
        # Regression line
        X = data[x_col].values.reshape(-1, 1)
        y = data[y_col].values
        reg = LinearRegression().fit(X, y)
        
        x_range = np.linspace(data[x_col].min(), data[x_col].max(), 100)
        y_pred = reg.predict(x_range.reshape(-1, 1))
        r2 = r2_score(y, reg.predict(X))
        
        ax.plot(x_range, y_pred, 'r--', alpha=0.8, 
                label=f'Regression (R²={r2:.3f})')
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'{x_col} vs {y_col}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        return filepath


class AnalysisPipeline:
    """Main analysis pipeline."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.data_loader = DataLoader()
        self.validator = DataValidator()
        self.models = {}
        self.analyzers = {}
        self.visualizer = Visualizer(
            os.path.join(config.output_dir, 'plots')
        )
        
        self._setup_components()
    
    def _setup_components(self):
        """Setup analysis components based on configuration."""
        # Setup models
        if 'linear' in self.config.model_types:
            self.models['linear'] = LinearModel()
        
        # Setup analyzers
        if 'correlation' in self.config.analysis_types:
            self.analyzers['correlation'] = CorrelationAnalyzer()
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run the complete analysis pipeline."""
        print("Starting Generic Data Analysis Pipeline")
        print("=" * 50)
        
        # 1. Load data
        data = self.data_loader.load(self.config.data_source)
        
        # 2. Validate data
        validation_config = {
            'required_columns': self.config.required_columns
        }
        validation_result = self.validator.validate(data, validation_config)
        
        if not validation_result['is_valid']:
            raise ValueError(f"Data validation failed: {validation_result['errors']}")
        
        # 3. Clean data
        data = data.dropna(subset=self.config.required_columns)
        print(f"Cleaned data shape: {data.shape}")
        
        # 4. Run analyses
        results = {}
        
        for name, analyzer in self.analyzers.items():
            results[name] = analyzer.analyze(data)
        
        # 5. Run models
        if self.models:
            X = data[self.config.feature_columns]
            y = data[self.config.target_column]
            
            for name, model in self.models.items():
                model.fit(X, y)
                results[f'model_{name}'] = {
                    'metrics': model.get_metrics(X, y),
                    'predictions': model.predict(X)
                }
        
        # 6. Generate visualizations
        if self.config.enable_plots:
            plots = self._generate_plots(data)
            results['visualizations'] = plots
        
        return results
    
    def _generate_plots(self, data: pd.DataFrame) -> List[str]:
        """Generate visualization plots."""
        plots = []
        
        # Create scatter plots for each feature vs target
        for feature in self.config.feature_columns:
            if feature in data.columns:
                plot_path = self.visualizer.create_scatter_plot(
                    data, feature, self.config.target_column,
                    f'{feature}_vs_{self.config.target_column}.png'
                )
                plots.append(plot_path)
        
        return plots


# Example usage
if __name__ == "__main__":
    # Example configuration for any tabular data analysis
    config = AnalysisConfig(
        data_source='public.csv',
        target_column='Reimb',
        feature_columns=['Days', 'Miles', 'Receipts'],
        required_columns=['Days', 'Miles', 'Receipts', 'Reimb'],
        analysis_types=['correlation'],
        model_types=['linear'],
        output_dir='outputs',
        enable_plots=True
    )
    
    # Run analysis
    pipeline = AnalysisPipeline(config)
    results = pipeline.run_analysis()
    
    # Print summary
    print("\nAnalysis Summary:")
    
    if 'model_linear' in results:
        metrics = results['model_linear']['metrics']
        print(f"Linear Model R² Score: {metrics['r2_score']:.3f}")
        print(f"Linear Model MAE: {metrics['mae']:.2f}")
    
    if 'correlation' in results:
        significant_pairs = results['correlation']['significant_pairs']
        print(f"Significant Correlations: {len(significant_pairs)}")
        for pair in significant_pairs:
            vars_str = f"{pair['variables'][0]} vs {pair['variables'][1]}"
            print(f"  {vars_str}: r={pair['correlation']:.3f}")
    
    if 'visualizations' in results:
        print(f"Plots generated: {len(results['visualizations'])}")
        for plot in results['visualizations']:
            print(f"  - {plot}")
    
    print("\nThis demonstrates a domain-agnostic analysis pipeline!")
    print("Change the configuration to work with any tabular data.") 