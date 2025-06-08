"""
Reporter

Generates analysis reports in various formats.
"""

import json
from pathlib import Path
from typing import Dict, Any

from core.interfaces import IReporter


class Reporter(IReporter):
    """Generates analysis reports."""
    
    def __init__(self, config: Dict[str, Any], output_dir: str = 'outputs/reports'):
        """
        Initialize reporter.
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory for reports
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, data, analysis_results: Dict[str, Any],
                       model_results: Dict[str, Any]) -> str:
        """
        Generate analysis report.
        
        Args:
            data: Source DataFrame
            analysis_results: Results from analyzers
            model_results: Results from models
            
        Returns:
            Path to generated report file
        """
        report_data = {
            'data_summary': {
                'total_records': len(data),
                'columns': list(data.columns)
            },
            'analysis_results': analysis_results,
            'model_results': model_results
        }
        
        # Generate JSON report
        report_file = self.output_dir / 'analysis_report.json'
        with open(report_file, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return str(report_file) 