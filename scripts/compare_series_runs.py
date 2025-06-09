#!/usr/bin/env python3
"""
Series Comparison Script

This script compares results across multiple runs in an analysis series,
providing insights into performance trends and parameter sensitivity.
"""

import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.utils import list_series_runs, get_latest_series_run


def load_run_results(run_dir: Path) -> dict:
    """Load results from a run directory."""
    results_file = run_dir / "results.json"
    if not results_file.exists():
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)


def compare_daily_rate_series(series_name: str, series_id: str = None) -> pd.DataFrame:
    """Compare results across runs in a daily rate analysis series."""
    
    runs = list_series_runs(series_name, series_id)
    if not runs:
        print(f"❌ No runs found for series: {series_name}")
        return None
    
    print(f"📊 Comparing {len(runs)} runs in series: {series_name}")
    print("=" * 60)
    
    comparison_data = []
    
    for run_dir in runs:
        results = load_run_results(run_dir)
        if not results:
            continue
        
        # Extract key metrics
        config = results['analysis_overview']['configuration']
        subregion_analysis = results['subregion_analysis']
        
        # Calculate aggregate metrics
        valid_subregions = [r for r in subregion_analysis.values() 
                          if isinstance(r, dict) and 'performance' in r]
        
        if valid_subregions:
            r2_scores = [r['performance']['r2_score'] for r in valid_subregions]
            rmse_scores = [r['performance']['rmse'] for r in valid_subregions]
            sample_counts = [r['count'] for r in valid_subregions]
            
            run_data = {
                'run': run_dir.name,
                'run_number': int(run_dir.name.split('_')[1]),
                'timestamp': results['analysis_overview']['timestamp'],
                'miles_threshold': config['miles_per_day_threshold'],
                'receipts_threshold': config['receipts_per_day_threshold'],
                'max_days': config['max_days'],
                'total_subregions': len(subregion_analysis),
                'valid_subregions': len(valid_subregions),
                'avg_r2': sum(r2_scores) / len(r2_scores),
                'max_r2': max(r2_scores),
                'min_r2': min(r2_scores),
                'avg_rmse': sum(rmse_scores) / len(rmse_scores),
                'min_rmse': min(rmse_scores),
                'max_rmse': max(rmse_scores),
                'total_samples': sum(sample_counts),
                'avg_samples_per_subregion': sum(sample_counts) / len(sample_counts)
            }
            
            comparison_data.append(run_data)
    
    if not comparison_data:
        print("❌ No valid results found in any runs")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('run_number')
    
    # Print comparison table
    print(f"\n{'Run':<8} {'Miles':<6} {'Receipts':<8} {'Valid':<6} {'Avg R²':<8} {'Max R²':<8} {'Avg RMSE':<10} {'Samples':<8}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        print(f"{row['run']:<8} {row['miles_threshold']:<6} ${row['receipts_threshold']:<7} "
              f"{row['valid_subregions']:<6} {row['avg_r2']:<8.3f} {row['max_r2']:<8.3f} "
              f"{row['avg_rmse']:<10.1f} {row['total_samples']:<8}")
    
    # Find best performing run
    best_run = df.loc[df['avg_r2'].idxmax()]
    print(f"\n🏆 Best performing run: {best_run['run']} (Avg R² = {best_run['avg_r2']:.3f})")
    
    return df


def create_series_comparison_plots(df: pd.DataFrame, series_name: str, output_dir: str = "outputs"):
    """Create visualization plots comparing series runs."""
    
    if df is None or len(df) == 0:
        return
    
    # Create output directory
    plot_dir = Path(output_dir) / f"{series_name}_comparison"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Series Comparison: {series_name}', fontsize=16, fontweight='bold')
    
    # 1. R² Score Trends
    ax1 = axes[0, 0]
    ax1.plot(df['run_number'], df['avg_r2'], 'o-', linewidth=2, markersize=8, label='Average R²')
    ax1.plot(df['run_number'], df['max_r2'], 's--', linewidth=2, markersize=6, label='Maximum R²')
    ax1.plot(df['run_number'], df['min_r2'], '^--', linewidth=2, markersize=6, label='Minimum R²')
    ax1.set_xlabel('Run Number')
    ax1.set_ylabel('R² Score')
    ax1.set_title('R² Score Trends Across Runs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. RMSE Trends
    ax2 = axes[0, 1]
    ax2.plot(df['run_number'], df['avg_rmse'], 'o-', linewidth=2, markersize=8, label='Average RMSE')
    ax2.plot(df['run_number'], df['min_rmse'], 's--', linewidth=2, markersize=6, label='Minimum RMSE')
    ax2.plot(df['run_number'], df['max_rmse'], '^--', linewidth=2, markersize=6, label='Maximum RMSE')
    ax2.set_xlabel('Run Number')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE Trends Across Runs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Valid Subregions
    ax3 = axes[0, 2]
    bars = ax3.bar(df['run_number'], df['valid_subregions'], alpha=0.7, color='skyblue')
    ax3.set_xlabel('Run Number')
    ax3.set_ylabel('Valid Subregions')
    ax3.set_title('Valid Subregions per Run')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, df['valid_subregions']):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{int(value)}', ha='center', va='bottom')
    
    # 4. Parameter Sensitivity (if parameters vary)
    ax4 = axes[1, 0]
    if df['miles_threshold'].nunique() > 1 or df['receipts_threshold'].nunique() > 1:
        scatter = ax4.scatter(df['miles_threshold'], df['receipts_threshold'], 
                            c=df['avg_r2'], s=df['total_samples']/10, 
                            alpha=0.7, cmap='viridis')
        ax4.set_xlabel('Miles Threshold')
        ax4.set_ylabel('Receipts Threshold')
        ax4.set_title('Parameter Sensitivity (size=samples, color=R²)')
        plt.colorbar(scatter, ax=ax4, label='Average R²')
    else:
        ax4.text(0.5, 0.5, 'Parameters constant\nacross runs', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Parameter Sensitivity')
    
    # 5. Sample Distribution
    ax5 = axes[1, 1]
    ax5.plot(df['run_number'], df['total_samples'], 'o-', linewidth=2, markersize=8, color='orange')
    ax5.set_xlabel('Run Number')
    ax5.set_ylabel('Total Samples')
    ax5.set_title('Sample Count per Run')
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance vs Samples
    ax6 = axes[1, 2]
    ax6.scatter(df['total_samples'], df['avg_r2'], s=100, alpha=0.7, color='red')
    ax6.set_xlabel('Total Samples')
    ax6.set_ylabel('Average R²')
    ax6.set_title('Performance vs Sample Size')
    ax6.grid(True, alpha=0.3)
    
    # Add run labels
    for i, row in df.iterrows():
        ax6.annotate(row['run'], (row['total_samples'], row['avg_r2']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = plot_dir / 'series_comparison.png'
    plt.savefig(plot_file, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Comparison plots saved to: {plot_file}")
    
    return plot_file


def generate_series_report(df: pd.DataFrame, series_name: str, output_dir: str = "outputs"):
    """Generate a comprehensive series comparison report."""
    
    if df is None or len(df) == 0:
        return
    
    report_dir = Path(output_dir) / f"{series_name}_comparison"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate report
    report = {
        'series_overview': {
            'series_name': series_name,
            'total_runs': len(df),
            'analysis_period': {
                'first_run': df['timestamp'].min(),
                'last_run': df['timestamp'].max()
            },
            'parameter_ranges': {
                'miles_threshold': {
                    'min': df['miles_threshold'].min(),
                    'max': df['miles_threshold'].max(),
                    'unique_values': df['miles_threshold'].unique().tolist()
                },
                'receipts_threshold': {
                    'min': df['receipts_threshold'].min(),
                    'max': df['receipts_threshold'].max(),
                    'unique_values': df['receipts_threshold'].unique().tolist()
                }
            }
        },
        'performance_summary': {
            'r2_statistics': {
                'overall_avg': df['avg_r2'].mean(),
                'overall_std': df['avg_r2'].std(),
                'best_run': {
                    'run': df.loc[df['avg_r2'].idxmax(), 'run'],
                    'value': df['avg_r2'].max()
                },
                'worst_run': {
                    'run': df.loc[df['avg_r2'].idxmin(), 'run'],
                    'value': df['avg_r2'].min()
                }
            },
            'rmse_statistics': {
                'overall_avg': df['avg_rmse'].mean(),
                'overall_std': df['avg_rmse'].std(),
                'best_run': {
                    'run': df.loc[df['avg_rmse'].idxmin(), 'run'],
                    'value': df['avg_rmse'].min()
                },
                'worst_run': {
                    'run': df.loc[df['avg_rmse'].idxmax(), 'run'],
                    'value': df['avg_rmse'].max()
                }
            }
        },
        'run_details': df.to_dict('records'),
        'trends': {
            'r2_trend': 'improving' if df['avg_r2'].iloc[-1] > df['avg_r2'].iloc[0] else 'declining',
            'rmse_trend': 'improving' if df['avg_rmse'].iloc[-1] < df['avg_rmse'].iloc[0] else 'declining',
            'sample_trend': 'increasing' if df['total_samples'].iloc[-1] > df['total_samples'].iloc[0] else 'decreasing'
        }
    }
    
    # Save report
    report_file = report_dir / 'series_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📋 Series report saved to: {report_file}")
    
    return report


def main():
    """Main function to demonstrate series comparison."""
    
    print("Series Comparison Tool")
    print("=" * 50)
    
    # Example: Compare daily rate analysis series
    series_name = "daily_rate_m150_r150"
    
    print(f"Analyzing series: {series_name}")
    
    # Load and compare results
    df = compare_daily_rate_series(series_name)
    
    if df is not None:
        # Create comparison plots
        plot_file = create_series_comparison_plots(df, series_name)
        
        # Generate comprehensive report
        report = generate_series_report(df, series_name)
        
        print(f"\n✅ Series comparison completed successfully!")
        print(f"📊 Analyzed {len(df)} runs")
        print(f"📈 Plots and reports saved to outputs/{series_name}_comparison/")
        
        # Show summary statistics
        print(f"\n📈 Performance Summary:")
        print(f"   Average R²: {df['avg_r2'].mean():.3f} ± {df['avg_r2'].std():.3f}")
        print(f"   Average RMSE: {df['avg_rmse'].mean():.1f} ± {df['avg_rmse'].std():.1f}")
        print(f"   Total Samples: {df['total_samples'].sum()}")
        
    else:
        print("❌ No series data found to compare")
        
        # List available series
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            series_dirs = [d for d in outputs_dir.iterdir() 
                          if d.is_dir() and '_20250608' in d.name]
            if series_dirs:
                print(f"\nAvailable series:")
                for series_dir in series_dirs:
                    runs = list(series_dir.glob("run_*"))
                    print(f"  - {series_dir.name} ({len(runs)} runs)")


if __name__ == "__main__":
    main() 