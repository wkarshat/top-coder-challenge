"""
Analysis Orchestrator

Main execution coordinator that manages the analysis pipeline,
handles configuration, and orchestrates data loading, analysis, and output.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.utils import load_config, setup_logging, ensure_directories, create_timestamped_output_dir
from core.loader import DataLoader
from analysis.models.linear_model import LinearModel
from analysis.models.ensemble import EnsembleModel, AdvancedEnsembleModel
from analysis.analyzers.correlation import CorrelationAnalyzer
from analysis.analyzers.statistical import StatisticalAnalyzer
from analysis.analyzers.advanced_stats import AdvancedStatisticalAnalyzer
from analysis.analyzers.time_series import TimeSeriesAnalyzer
from analysis.analyzers.clustering import ClusteringAnalyzer
from analysis.output.visualizer import Visualizer
from analysis.output.reporter import Reporter

logger = logging.getLogger(__name__)


class AnalysisOrchestrator:
    """Main orchestrator for analysis pipeline execution."""
    
    def __init__(self, config_path: str = 'config.yaml', 
                 analysis_config_path: str = 'analysis.yaml'):
        """
        Initialize orchestrator with configuration.
        
        Args:
            config_path: Path to system configuration file
            analysis_config_path: Path to analysis configuration file
        """
        self.config = load_config(config_path)
        self.analysis_config = load_config(analysis_config_path)
        
        self.setup_environment()
        self.initialize_components()
    
    def setup_environment(self):
        """Set up logging, directories, and environment."""
        # Setup logging
        setup_logging(self.config.get('logging', {}))
        logger.info("Analysis orchestrator initialized")
        
        # Create timestamped output directory
        output_config = self.config.get('output', {})
        base_dir = output_config.get('base_dir', 'outputs')
        self.output_dir = create_timestamped_output_dir(base_dir, 'analysis')
        logger.info(f"Created timestamped output directory: {self.output_dir}")
    
    def initialize_components(self):
        """Initialize analysis components."""
        self.loader = DataLoader()
        
        # Initialize models
        self.models = {}
        model_types = self.analysis_config.get('models', {}).get('types', [])
        for model_type in model_types:
            if model_type == 'linear':
                model_config = self.analysis_config['models'].get('linear', {})
                self.models['linear'] = LinearModel(**model_config)
            elif model_type == 'ensemble':
                model_config = self.analysis_config['models'].get('ensemble', {})
                self.models['ensemble'] = EnsembleModel(model_config)
            elif model_type == 'advanced_ensemble':
                model_config = self.analysis_config['models'].get('advanced_ensemble', {})
                self.models['advanced_ensemble'] = AdvancedEnsembleModel(model_config)
        
        # Initialize analyzers
        self.analyzers = {}
        analysis_types = self.analysis_config.get('analysis', {}).get('types', [])
        for analysis_type in analysis_types:
            if analysis_type == 'correlation':
                config = self.analysis_config['analysis'].get('correlation', {})
                self.analyzers['correlation'] = CorrelationAnalyzer(config)
            elif analysis_type == 'statistical':
                config = self.analysis_config['analysis'].get('statistical', {})
                self.analyzers['statistical'] = StatisticalAnalyzer(config)
            elif analysis_type == 'advanced_stats':
                config = self.analysis_config['analysis'].get('advanced_stats', {})
                self.analyzers['advanced_stats'] = AdvancedStatisticalAnalyzer(config)
            elif analysis_type == 'time_series':
                config = self.analysis_config['analysis'].get('time_series', {})
                self.analyzers['time_series'] = TimeSeriesAnalyzer(config)
            elif analysis_type == 'clustering':
                config = self.analysis_config['analysis'].get('clustering', {})
                self.analyzers['clustering'] = ClusteringAnalyzer(config)
        
        # Initialize output components (will be updated with timestamped paths)
        viz_config = self.analysis_config.get('visualization', {})
        self.visualizer = None  # Will be initialized in run_analysis
        
        report_config = self.config.get('output', {})
        self.reporter = None  # Will be initialized in run_analysis
        
        logger.info(f"Initialized {len(self.models)} models, "
                   f"{len(self.analyzers)} analyzers")
    
    def run_analysis(self, data_source: str) -> Dict[str, Any]:
        """
        Run complete analysis pipeline.
        
        Args:
            data_source: Path to data file
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info(f"Starting analysis pipeline for {data_source}")
        
        try:
            # 1. Load and process data
            data = self.loader.load_and_process(
                data_source, 
                self.analysis_config.get('data', {})
            )
            logger.info(f"Data loaded: {len(data)} records")
            
            # 2. Run analyses
            analysis_results = self._run_analyses(data)
            
            # 3. Run models
            model_results = self._run_models(data)
            
            # 4. Generate outputs
            output_results = self._generate_outputs(
                data, analysis_results, model_results
            )
            
            # Combine all results
            results = {
                'data_info': {
                    'source': data_source,
                    'records': len(data),
                    'columns': list(data.columns)
                },
                'analysis': analysis_results,
                'models': model_results,
                'outputs': output_results
            }
            
            logger.info("Analysis pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {e}")
            raise
    
    def _run_analyses(self, data) -> Dict[str, Any]:
        """Run all configured analyses."""
        if not self.analyzers:
            logger.info("No analyzers configured")
            return {}
        
        performance_config = self.config.get('performance', {})
        use_parallel = performance_config.get('parallel_processing', False)
        
        if use_parallel and len(self.analyzers) > 1:
            return self._run_analyses_parallel(data)
        else:
            return self._run_analyses_sequential(data)
    
    def _run_analyses_sequential(self, data) -> Dict[str, Any]:
        """Run analyses sequentially."""
        results = {}
        
        for name, analyzer in self.analyzers.items():
            logger.info(f"Running {name} analysis")
            try:
                results[name] = analyzer.analyze(data)
                logger.info(f"{name} analysis completed")
            except Exception as e:
                logger.error(f"{name} analysis failed: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def _run_analyses_parallel(self, data) -> Dict[str, Any]:
        """Run analyses in parallel."""
        results = {}
        max_workers = self.config.get('performance', {}).get('max_workers', 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all analysis tasks
            futures = {
                executor.submit(analyzer.analyze, data): name
                for name, analyzer in self.analyzers.items()
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                    logger.info(f"{name} analysis completed")
                except Exception as e:
                    logger.error(f"{name} analysis failed: {e}")
                    results[name] = {'error': str(e)}
        
        return results
    
    def _run_models(self, data) -> Dict[str, Any]:
        """Run all configured models."""
        if not self.models:
            logger.info("No models configured")
            return {}
        
        results = {}
        
        # Get target and feature columns
        data_config = self.analysis_config.get('data', {})
        target_col = data_config.get('target_column')
        feature_cols = data_config.get('feature_columns', [])
        
        if not target_col or not feature_cols:
            logger.warning("Target or feature columns not specified")
            return {}
        
        # Prepare data for modeling
        available_features = [col for col in feature_cols if col in data.columns]
        if not available_features:
            logger.warning("No feature columns found in data")
            return {}
        
        X = data[available_features]
        y = data[target_col]
        
        # Run each model
        for name, model in self.models.items():
            logger.info(f"Training {name} model")
            try:
                # Fit model
                model.fit(X, y)
                
                # Get predictions and metrics
                predictions = model.predict(X)
                metrics = model.get_metrics(y, predictions)
                
                results[name] = {
                    'metrics': metrics,
                    'predictions': predictions.tolist(),
                    'feature_columns': available_features,
                    'target_column': target_col
                }
                
                logger.info(f"{name} model completed")
                
            except Exception as e:
                logger.error(f"{name} model failed: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def _generate_outputs(self, data, analysis_results, 
                         model_results) -> Dict[str, Any]:
        """Generate visualizations and reports."""
        output_results = {}
        
        try:
            # Initialize output components with timestamped directories
            if self.visualizer is None:
                viz_config = self.analysis_config.get('visualization', {})
                plots_dir = f"{self.output_dir}/plots"
                self.visualizer = Visualizer(viz_config, plots_dir)
            
            if self.reporter is None:
                report_config = self.config.get('output', {})
                reports_dir = f"{self.output_dir}/reports"
                self.reporter = Reporter(report_config, reports_dir)
            
            # Generate visualizations
            logger.info("Generating visualizations")
            plot_files = self.visualizer.create_plots(
                data, analysis_results, model_results
            )
            output_results['plots'] = plot_files
            
            # Generate report
            logger.info("Generating report")
            report_file = self.reporter.generate_report(
                data, analysis_results, model_results
            )
            output_results['report'] = report_file
            
        except Exception as e:
            logger.error(f"Output generation failed: {e}")
            output_results['error'] = str(e)
        
        return output_results
    
    def run_batch_analysis(self, data_sources: List[str]) -> Dict[str, Any]:
        """
        Run analysis on multiple data sources.
        
        Args:
            data_sources: List of data file paths
            
        Returns:
            Dictionary with results for each data source
        """
        logger.info(f"Starting batch analysis for {len(data_sources)} sources")
        
        batch_results = {}
        
        for source in data_sources:
            logger.info(f"Processing {source}")
            try:
                results = self.run_analysis(source)
                batch_results[source] = results
            except Exception as e:
                logger.error(f"Failed to process {source}: {e}")
                batch_results[source] = {'error': str(e)}
        
        logger.info("Batch analysis completed")
        return batch_results


def print_summary(results: Dict[str, Any]):
    """Print a summary of analysis results."""
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    # Data info
    data_info = results.get('data_info', {})
    print(f"\nData Source: {data_info.get('source', 'Unknown')}")
    print(f"Records: {data_info.get('records', 0)}")
    print(f"Columns: {len(data_info.get('columns', []))}")
    
    # Analysis results
    analysis = results.get('analysis', {})
    if analysis:
        print(f"\nAnalyses Run: {len(analysis)}")
        for name, result in analysis.items():
            if 'error' in result:
                print(f"  {name}: ERROR - {result['error']}")
            else:
                print(f"  {name}: SUCCESS")
    
    # Model results
    models = results.get('models', {})
    if models:
        print(f"\nModels Run: {len(models)}")
        for name, result in models.items():
            if 'error' in result:
                print(f"  {name}: ERROR - {result['error']}")
            else:
                metrics = result.get('metrics', {})
                r2 = metrics.get('r2_score', 'N/A')
                print(f"  {name}: R² = {r2}")
    
    # Output files
    outputs = results.get('outputs', {})
    if outputs:
        print(f"\nOutputs Generated:")
        plots = outputs.get('plots', [])
        if plots:
            print(f"  Plots: {len(plots)} files")
        report = outputs.get('report')
        if report:
            print(f"  Report: {report}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Legacy Analysis System')
    parser.add_argument('--config', default='../config.yaml',
                        help='Path to system configuration file')
    parser.add_argument('--analysis-config', default='../analysis.yaml',
                        help='Path to analysis configuration file')
    parser.add_argument('--data', required=True,
                        help='Path to data file to analyze')
    parser.add_argument('--batch', action='store_true',
                        help='Run batch analysis on multiple files')
    
    args = parser.parse_args()
    
    try:
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(args.config, args.analysis_config)
        
        if args.batch:
            # Batch mode - treat data as list of files
            data_sources = [args.data]  # For now, single file
            results = orchestrator.run_batch_analysis(data_sources)
            
            # Print summary for each source
            for source, result in results.items():
                print(f"\n{'='*20} {source} {'='*20}")
                print_summary(result)
        else:
            # Single file mode
            results = orchestrator.run_analysis(args.data)
            print_summary(results)
            
            # Print output directory info
            print(f"\nAll outputs saved to: {orchestrator.output_dir}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc() 