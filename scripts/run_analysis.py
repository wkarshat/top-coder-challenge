#!/usr/bin/env python3
"""
Analysis Runner Script

Main entry point for running legacy analysis system.
Provides command-line interface with configuration options.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from core.main import AnalysisOrchestrator, print_summary
except ImportError:
    # Fallback for development
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from core.main import AnalysisOrchestrator, print_summary


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Legacy Analysis System - Modular Data Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python scripts/run_analysis.py --data-source public.csv
  
  # Custom configuration
  python scripts/run_analysis.py \\
      --data-source data/reimbursement.csv \\
      --config config.yaml \\
      --analysis-config analysis.yaml \\
      --output-dir results/
  
  # Batch processing
  python scripts/run_analysis.py \\
      --data-source public.csv,private.csv \\
      --batch
        """
    )
    
    parser.add_argument(
        '--data-source', '-d',
        required=True,
        help='Data source file(s). For batch processing, separate with commas'
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='System configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--analysis-config', '-a',
        default='analysis.yaml',
        help='Analysis configuration file (default: analysis.yaml)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        help='Output directory (overrides config setting)'
    )
    
    parser.add_argument(
        '--batch', '-b',
        action='store_true',
        help='Process multiple data sources (comma-separated)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output except errors'
    )
    
    return parser.parse_args()


def validate_files(args):
    """Validate that required files exist."""
    errors = []
    
    # Check configuration files
    if not Path(args.config).exists():
        errors.append(f"Config file not found: {args.config}")
    
    if not Path(args.analysis_config).exists():
        errors.append(f"Analysis config file not found: {args.analysis_config}")
    
    # Check data sources
    data_sources = args.data_source.split(',') if args.batch else [args.data_source]
    
    for source in data_sources:
        source = source.strip()
        if not Path(source).exists() and not Path(f'data/{source}').exists():
            errors.append(f"Data source not found: {source}")
    
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


def setup_logging_level(args):
    """Set up logging level based on arguments."""
    import logging
    
    if args.quiet:
        level = logging.ERROR
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set up logging
    setup_logging_level(args)
    
    # Validate files
    if not validate_files(args):
        sys.exit(1)
    
    try:
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(
            config_path=args.config,
            analysis_config_path=args.analysis_config
        )
        
        # Override output directory if specified
        if args.output_dir:
            orchestrator.config['output']['base_dir'] = args.output_dir
        
        # Run analysis
        if args.batch:
            # Batch processing
            data_sources = [s.strip() for s in args.data_source.split(',')]
            
            if not args.quiet:
                print(f"Running batch analysis on {len(data_sources)} sources...")
            
            results = orchestrator.run_batch_analysis(data_sources)
            
            # Print summary for each source
            if not args.quiet:
                for source, result in results.items():
                    print(f"\n{'='*20} {source} {'='*20}")
                    if 'error' in result:
                        print(f"ERROR: {result['error']}")
                    else:
                        print_summary(result)
        
        else:
            # Single source processing
            if not args.quiet:
                print(f"Running analysis on {args.data_source}...")
            
            results = orchestrator.run_analysis(args.data_source)
            
            # Print summary
            if not args.quiet:
                print_summary(results)
        
        if not args.quiet:
            print("\nAnalysis completed successfully!")
    
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 