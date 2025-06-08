"""
Enhanced Linear Regression Analysis for Legacy Reimbursement System

This module applies linear regression to the legacy reimbursement data with 
enhanced features:
- Command line options for CSV file selection
- Analysis by days (1-14) with correlations and regressions
- Visualization of Miles vs Receipts with Reimb on secondary axis
- Configurable day ranges
- Comprehensive data visualization
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for faster rendering
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')  # Use simpler style for faster rendering


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Linear Regression Analysis for Legacy Reimbursement')
    parser.add_argument('--csv', '-c', default='public.csv', 
                        help='Input CSV file name (default: public.csv)')
    parser.add_argument('--days', '-d', default='all',
                        help='Days to process: "all" (1-14), range "1-5", '
                             'or list "1,3,5,7" (default: all)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip individual day plots for faster execution')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: auto-generated timestamped)')
    return parser.parse_args()


def create_output_directory(base_dir='outputs'):
    """Create timestamped output directory for linear regression analysis."""
    timestamp = datetime.now().strftime('%H%M%S')
    output_dir = Path(f"{base_dir}/linear_regression_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def parse_days_argument(days_arg, available_days):
    """Parse the days argument to return list of days to process."""
    if days_arg.lower() == 'all':
        return sorted(available_days)
    
    if '-' in days_arg:
        # Range format: "1-5"
        start, end = map(int, days_arg.split('-'))
        return [d for d in range(start, end + 1) if d in available_days]
    
    if ',' in days_arg:
        # List format: "1,3,5,7"
        requested_days = [int(d.strip()) for d in days_arg.split(',')]
        return [d for d in requested_days if d in available_days]
    
    # Single day
    day = int(days_arg)
    return [day] if day in available_days else []


def load_data(csv_file):
    """Load the reimbursement data from specified CSV file."""
    try:
        df = pd.read_csv(csv_file)
        print(f"Data loaded successfully from {csv_file}")
        print(f"Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found. Trying data/{csv_file}")
        df = pd.read_csv(f'data/{csv_file}')
        print(f"Data loaded successfully from data/{csv_file}")
    
    # Convert columns to numeric, handling any formatting issues
    numeric_cols = ['Days', 'Miles', 'Receipts', 'Reimb']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing values in key columns
    df = df.dropna(subset=numeric_cols)
    
    print(f"Data shape after cleaning: {df.shape}")
    print(f"Days range: {df['Days'].min()} to {df['Days'].max()}")
    print(f"Available days: {sorted(df['Days'].unique())}")
    
    return df


def calculate_correlations_and_regression(df_day):
    """Calculate correlations and regression for a specific day's data."""
    if len(df_day) < 3:  # Need at least 3 points for meaningful analysis
        return None
    
    # Overall correlations
    miles_receipts_corr, miles_receipts_p = pearsonr(df_day['Miles'], 
                                                     df_day['Receipts'])
    miles_reimb_corr, miles_reimb_p = pearsonr(df_day['Miles'], 
                                               df_day['Reimb'])
    receipts_reimb_corr, receipts_reimb_p = pearsonr(df_day['Receipts'], 
                                                     df_day['Reimb'])
    
    # Binned correlations
    binned_correlations = calculate_binned_correlations(df_day)
    
    # Linear regression: Miles vs Receipts
    X_miles = df_day['Miles'].values.reshape(-1, 1)
    y_receipts = df_day['Receipts'].values
    reg_miles_receipts = LinearRegression().fit(X_miles, y_receipts)
    receipts_pred = reg_miles_receipts.predict(X_miles)
    r2_miles_receipts = r2_score(y_receipts, receipts_pred)
    
    # Linear regression: Miles vs Reimb
    y_reimb = df_day['Reimb'].values
    reg_miles_reimb = LinearRegression().fit(X_miles, y_reimb)
    reimb_pred = reg_miles_reimb.predict(X_miles)
    r2_miles_reimb = r2_score(y_reimb, reimb_pred)
    
    return {
        'correlations': {
            'miles_receipts': (miles_receipts_corr, miles_receipts_p),
            'miles_reimb': (miles_reimb_corr, miles_reimb_p),
            'receipts_reimb': (receipts_reimb_corr, receipts_reimb_p)
        },
        'binned_correlations': binned_correlations,
        'regressions': {
            'miles_receipts': {
                'model': reg_miles_receipts,
                'r2': r2_miles_receipts,
                'predictions': receipts_pred
            },
            'miles_reimb': {
                'model': reg_miles_reimb,
                'r2': r2_miles_reimb,
                'predictions': reimb_pred
            }
        }
    }


def calculate_binned_correlations(df_day):
    """Calculate correlations within bins for Miles and Receipts with Reimb."""
    binned_results = {
        'miles_bins': {},
        'receipts_bins': {}
    }
    
    # Miles binning (increments of 100)
    miles_min = int(df_day['Miles'].min() // 100) * 100
    miles_max = int(df_day['Miles'].max() // 100 + 1) * 100
    
    for bin_start in range(miles_min, miles_max, 100):
        bin_end = bin_start + 100
        bin_data = df_day[(df_day['Miles'] >= bin_start) & 
                         (df_day['Miles'] < bin_end)]
        
        if len(bin_data) >= 3:  # Need at least 3 points for correlation
            corr, p_val = pearsonr(bin_data['Miles'], bin_data['Reimb'])
            binned_results['miles_bins'][f'{bin_start}-{bin_end}'] = {
                'correlation': corr,
                'p_value': p_val,
                'count': len(bin_data),
                'mean_miles': bin_data['Miles'].mean(),
                'mean_reimb': bin_data['Reimb'].mean()
            }
    
    # Receipts binning (increments of 200)
    receipts_min = int(df_day['Receipts'].min() // 200) * 200
    receipts_max = int(df_day['Receipts'].max() // 200 + 1) * 200
    
    for bin_start in range(receipts_min, receipts_max, 200):
        bin_end = bin_start + 200
        bin_data = df_day[(df_day['Receipts'] >= bin_start) & 
                         (df_day['Receipts'] < bin_end)]
        
        if len(bin_data) >= 3:  # Need at least 3 points for correlation
            corr, p_val = pearsonr(bin_data['Receipts'], bin_data['Reimb'])
            binned_results['receipts_bins'][f'{bin_start}-{bin_end}'] = {
                'correlation': corr,
                'p_value': p_val,
                'count': len(bin_data),
                'mean_receipts': bin_data['Receipts'].mean(),
                'mean_reimb': bin_data['Reimb'].mean()
            }
    
    return binned_results


def create_day_analysis_plot(df_day, day, analysis_results, output_dir='.'):
    """Create two separate plots for a specific day's analysis."""
    plot_files = []
    
    # Plot 1: Miles vs Reimb
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Scatter plot: Miles vs Reimb
    ax1.scatter(df_day['Miles'], df_day['Reimb'], 
                c='red', alpha=0.6, s=30, label='Data Points')
    
    # Regression line for Miles vs Reimb
    if analysis_results and 'regressions' in analysis_results:
        miles_sorted = np.sort(df_day['Miles'])
        reimb_pred_sorted = analysis_results['regressions'][
            'miles_reimb']['model'].predict(miles_sorted.reshape(-1, 1))
        r2_reimb = analysis_results['regressions']['miles_reimb']['r2']
        corr_reimb = analysis_results['correlations']['miles_reimb'][0]
        ax1.plot(miles_sorted, reimb_pred_sorted, 'r--', alpha=0.8,
                 label=f"Regression (R²={r2_reimb:.3f}, r={corr_reimb:.3f})")
    
    ax1.set_xlabel('Miles')
    ax1.set_ylabel('Reimb')
    ax1.set_title(f'Day {day}: Miles vs Reimb (n={len(df_day)})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot1_file = Path(output_dir) / f'day_{day}_miles_vs_reimb.png'
    plt.savefig(plot1_file, dpi=100, bbox_inches='tight')
    plt.close()
    plot_files.append(str(plot1_file))
    
    # Plot 2: Receipts vs Reimb
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Scatter plot: Receipts vs Reimb
    ax2.scatter(df_day['Receipts'], df_day['Reimb'], 
                c='blue', alpha=0.6, s=30, label='Data Points')
    
    # Regression line for Receipts vs Reimb (we need to create this regression)
    if analysis_results and len(df_day) >= 3:
        X_receipts = df_day['Receipts'].values.reshape(-1, 1)
        y_reimb = df_day['Reimb'].values
        reg_receipts_reimb = LinearRegression().fit(X_receipts, y_reimb)
        receipts_pred = reg_receipts_reimb.predict(X_receipts)
        r2_receipts_reimb = r2_score(y_reimb, receipts_pred)
        
        receipts_sorted = np.sort(df_day['Receipts'])
        reimb_pred_sorted = reg_receipts_reimb.predict(
            receipts_sorted.reshape(-1, 1))
        corr_receipts_reimb = analysis_results['correlations']['receipts_reimb'][0]
        ax2.plot(receipts_sorted, reimb_pred_sorted, 'b--', alpha=0.8,
                 label=f"Regression (R²={r2_receipts_reimb:.3f}, r={corr_receipts_reimb:.3f})")
    
    ax2.set_xlabel('Receipts')
    ax2.set_ylabel('Reimb')
    ax2.set_title(f'Day {day}: Receipts vs Reimb (n={len(df_day)})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot2_file = Path(output_dir) / f'day_{day}_receipts_vs_reimb.png'
    plt.savefig(plot2_file, dpi=100, bbox_inches='tight')
    plt.close()
    plot_files.append(str(plot2_file))
    
    return plot_files


def create_summary_visualization(df, day_results, output_dir='.'):
    """Create simplified summary visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Distribution of data by days
    day_counts = df['Days'].value_counts().sort_index()
    ax1.bar(day_counts.index, day_counts.values, alpha=0.7)
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Count')
    ax1.set_title('Data Distribution by Days')
    
    # 2. Miles vs Receipts colored by Days
    scatter = ax2.scatter(df['Miles'], df['Receipts'], c=df['Days'], 
                         cmap='viridis', alpha=0.6, s=20)
    ax2.set_xlabel('Miles')
    ax2.set_ylabel('Receipts')
    ax2.set_title('Miles vs Receipts (colored by Days)')
    plt.colorbar(scatter, ax=ax2, label='Days')
    
    # 3. R² values by days
    if day_results:
        days_with_results = [day for day, result in day_results.items() 
                           if result is not None]
        if days_with_results:
            r2_miles_receipts = [day_results[day]['regressions'][
                'miles_receipts']['r2'] for day in days_with_results]
            r2_miles_reimb = [day_results[day]['regressions'][
                'miles_reimb']['r2'] for day in days_with_results]
            
            x = np.arange(len(days_with_results))
            width = 0.35
            ax3.bar(x - width/2, r2_miles_receipts, width, 
                   label='Miles→Receipts', alpha=0.7)
            ax3.bar(x + width/2, r2_miles_reimb, width, 
                   label='Miles→Reimb', alpha=0.7)
            ax3.set_xlabel('Days')
            ax3.set_ylabel('R² Score')
            ax3.set_title('Regression R² by Day')
            ax3.set_xticks(x)
            ax3.set_xticklabels([f'Day {d}' for d in days_with_results])
            ax3.legend()
    
    # 4. Box plots by days
    days_for_box = sorted(df['Days'].unique())
    reimb_by_day = [df[df['Days'] == day]['Reimb'].values 
                   for day in days_for_box]
    ax4.boxplot(reimb_by_day, labels=[f'D{d}' for d in days_for_box])
    ax4.set_xlabel('Days')
    ax4.set_ylabel('Reimb')
    ax4.set_title('Reimb Distribution by Days')
    
    plt.tight_layout()
    summary_file = Path(output_dir) / 'summary_analysis.png'
    plt.savefig(summary_file, dpi=100, bbox_inches='tight')
    plt.close()  # Close to free memory
    return str(summary_file)


def print_day_binned_analysis(day, df_day, analysis_results):
    """Print detailed binned correlation analysis for a specific day."""
    print(f"\n{'='*60}")
    print(f"DAY {day} DETAILED BINNED CORRELATION ANALYSIS")
    print(f"{'='*60}")
    print(f"Total Records: {len(df_day)}")
    
    if analysis_results and 'binned_correlations' in analysis_results:
        binned = analysis_results['binned_correlations']
        
        # Miles-Reimb correlations by bins
        print(f"\nMILES-REIMB CORRELATIONS (100-mile bins):")
        print(f"{'Bin Range':>15} {'Count':>6} {'Correlation':>12} {'P-value':>10} "
              f"{'Mean Miles':>12} {'Mean Reimb':>12}")
        print("-" * 80)
        
        for bin_range, data in sorted(binned['miles_bins'].items()):
            print(f"{bin_range:>15} {data['count']:>6} {data['correlation']:>12.3f} "
                  f"{data['p_value']:>10.3f} {data['mean_miles']:>12.1f} "
                  f"{data['mean_reimb']:>12.1f}")
        
        # Receipts-Reimb correlations by bins
        print(f"\nRECEIPTS-REIMB CORRELATIONS (200-receipt bins):")
        print(f"{'Bin Range':>15} {'Count':>6} {'Correlation':>12} {'P-value':>10} "
              f"{'Mean Receipts':>14} {'Mean Reimb':>12}")
        print("-" * 82)
        
        for bin_range, data in sorted(binned['receipts_bins'].items()):
            print(f"{bin_range:>15} {data['count']:>6} {data['correlation']:>12.3f} "
                  f"{data['p_value']:>10.3f} {data['mean_receipts']:>14.1f} "
                  f"{data['mean_reimb']:>12.1f}")
        
        # Overall correlations for comparison
        overall_corr = analysis_results['correlations']
        print(f"\nOVERALL CORRELATIONS (for comparison):")
        print(f"Miles-Reimb: {overall_corr['miles_reimb'][0]:.3f} "
              f"(p={overall_corr['miles_reimb'][1]:.3f})")
        print(f"Receipts-Reimb: {overall_corr['receipts_reimb'][0]:.3f} "
              f"(p={overall_corr['receipts_reimb'][1]:.3f})")


def print_summary_statistics(df, day_results):
    """Print comprehensive summary statistics."""
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*80)
    
    print("\nDataset Overview:")
    print(f"Total records: {len(df)}")
    print(f"Days range: {df['Days'].min()} to {df['Days'].max()}")
    print(f"Unique days: {len(df['Days'].unique())}")
    
    print("\nOverall Statistics:")
    for col in ['Miles', 'Receipts', 'Reimb']:
        print(f"{col:>10}: μ={df[col].mean():8.2f}, "
              f"σ={df[col].std():8.2f}, "
              f"range=[{df[col].min():8.2f}, {df[col].max():8.2f}]")
    
    if day_results:
        print("\nDay-by-Day Analysis:")
        print(f"{'Day':>4} {'N':>4} {'Miles-Receipts':>15} "
              f"{'Miles-Reimb':>12} {'Receipts-Reimb':>14} "
              f"{'R²(M→R)':>8} {'R²(M→Re)':>9}")
        print("-" * 85)
        
        for day in sorted(day_results.keys()):
            result = day_results[day]
            if result is not None:
                n = len(df[df['Days'] == day])
                mr_corr = result['correlations']['miles_receipts'][0]
                mre_corr = result['correlations']['miles_reimb'][0]
                rre_corr = result['correlations']['receipts_reimb'][0]
                r2_mr = result['regressions']['miles_receipts']['r2']
                r2_mre = result['regressions']['miles_reimb']['r2']
                
                print(f"{day:>4} {n:>4} {mr_corr:>15.3f} {mre_corr:>12.3f} "
                      f"{rre_corr:>14.3f} {r2_mr:>8.3f} {r2_mre:>9.3f}")


def main():
    """Main function to run the enhanced linear regression analysis."""
    args = parse_arguments()
    
    print("Enhanced Linear Regression Analysis")
    print("=" * 50)
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_output_directory()
    
    print(f"Output directory: {output_dir}")
    
    # Load data
    df = load_data(args.csv)
    
    # Parse days to process
    available_days = sorted(df['Days'].unique())
    days_to_process = parse_days_argument(args.days, available_days)
    
    if not days_to_process:
        print(f"Error: No valid days found for argument '{args.days}'")
        print(f"Available days: {available_days}")
        return
    
    print(f"\nProcessing days: {days_to_process}")
    
    # Analyze each day
    day_results = {}
    all_plot_files = []
    
    for day in days_to_process:
        print(f"\nAnalyzing Day {day}...")
        df_day = df[df['Days'] == day].copy()
        
        if len(df_day) == 0:
            print(f"  No data for day {day}")
            day_results[day] = None
            continue
        
        print(f"  Records: {len(df_day)}")
        
        # Calculate correlations and regressions
        analysis_results = calculate_correlations_and_regression(df_day)
        day_results[day] = analysis_results
        
        if analysis_results is None:
            print("  Insufficient data for analysis (need at least 3 records)")
            continue
        
        # Print detailed binned analysis for this day
        print_day_binned_analysis(day, df_day, analysis_results)
        
        # Create plots for this day (if not skipped)
        if not args.no_plots:
            plot_files = create_day_analysis_plot(df_day, day, analysis_results, output_dir)
            all_plot_files.extend(plot_files)
            print(f"  Plots saved: {', '.join([Path(f).name for f in plot_files])}")
        
        # Print day summary
        corr = analysis_results['correlations']
        reg = analysis_results['regressions']
        print(f"\nDay {day} Summary:")
        print(f"  Miles↔Receipts: r={corr['miles_receipts'][0]:.3f}, "
              f"R²={reg['miles_receipts']['r2']:.3f}")
        print(f"  Miles↔Reimb: r={corr['miles_reimb'][0]:.3f}, "
              f"R²={reg['miles_reimb']['r2']:.3f}")
        print(f"  Receipts↔Reimb: r={corr['receipts_reimb'][0]:.3f}")
    
    # Create summary visualization
    print("\nCreating summary visualization...")
    summary_file = create_summary_visualization(df, day_results, output_dir)
    print(f"Summary plot saved: {Path(summary_file).name}")
    
    # Print summary statistics
    print_summary_statistics(df, day_results)
    
    processed_days = len([r for r in day_results.values() if r is not None])
    print(f"\nAnalysis complete! Processed {processed_days} days with "
          f"sufficient data.")
    
    if all_plot_files:
        print(f"\nGenerated {len(all_plot_files)} individual day plots:")
        for i, plot_file in enumerate(all_plot_files, 1):
            print(f"  {i:2d}. {Path(plot_file).name}")
    print(f"\nSummary plot: {Path(summary_file).name}")
    print(f"\nTotal plots generated: {len(all_plot_files) + 1}")
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main() 