"""
Visualizer

Creates plots and visualizations for analysis results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List

from core.interfaces import IVisualizer


class Visualizer(IVisualizer):
    """Creates visualizations for analysis results."""
    
    def __init__(self, config: Dict[str, Any], output_dir: str = 'outputs'):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory for plots
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib backend for non-interactive use
        plt.switch_backend('Agg')
    
    def create_plots(self, data: pd.DataFrame, analysis_results: Dict[str, Any],
                    model_results: Dict[str, Any]) -> List[str]:
        """
        Create plots based on configuration and results.
        
        Args:
            data: Source DataFrame
            analysis_results: Results from analyzers
            model_results: Results from models
            
        Returns:
            List of created plot file paths
        """
        plot_files = []
        
        plots_config = self.config.get('plots', [])
        
        for plot_config in plots_config:
            plot_type = plot_config.get('type')
            
            try:
                if plot_type == 'scatter':
                    file_path = self._create_scatter_plot(data, plot_config)
                    if file_path:
                        plot_files.append(file_path)
                
                elif plot_type == 'correlation_heatmap':
                    file_path = self._create_correlation_heatmap(
                        analysis_results, plot_config
                    )
                    if file_path:
                        plot_files.append(file_path)
                
            except Exception as e:
                print(f"Failed to create {plot_type} plot: {e}")
        
        return plot_files
    
    def _create_scatter_plot(self, data: pd.DataFrame, 
                           config: Dict[str, Any]) -> str:
        """Create a scatter plot."""
        x_col = config.get('x')
        y_col = config.get('y')
        
        if not x_col or not y_col:
            return None
        
        if x_col not in data.columns or y_col not in data.columns:
            return None
        
        plt.figure(figsize=(10, 6))
        plt.scatter(data[x_col], data[y_col], alpha=0.6)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'{x_col} vs {y_col}')
        plt.grid(True, alpha=0.3)
        
        filename = f'scatter_{x_col}_vs_{y_col}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_correlation_heatmap(self, analysis_results: Dict[str, Any],
                                  config: Dict[str, Any]) -> str:
        """Create a correlation heatmap."""
        correlation_data = analysis_results.get('correlation', {})
        corr_matrix = correlation_data.get('correlation_matrix')
        
        if not corr_matrix:
            return None
        
        # Convert dict back to DataFrame
        corr_df = pd.DataFrame(corr_matrix)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        filename = 'correlation_heatmap.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        return str(filepath) 