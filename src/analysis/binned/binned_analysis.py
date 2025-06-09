#!/usr/bin/env python3
"""
Binned Analysis Across Parameter Half-Ranges

Divides the parameter space into 8 regions based on half-ranges:
- Days: 1-15 (low) vs 16-30 (high)
- Miles: 0-500 (low) vs 501-1000 (high)  
- Receipts: 0-500 (low) vs 501-1000 (high)

Creates 2³ = 8 combinations for detailed pattern analysis.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_and_bin_data(csv_file):
    """Load data and create binned categories."""
    df = pd.read_csv(csv_file)
    
    # Convert to numeric
    for col in ['Days', 'Miles', 'Receipts', 'Reimb']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    # Create binary bins based on half-ranges
    df['Days_Bin'] = (df['Days'] > 15).astype(int)  # 0: 1-15, 1: 16-30
    df['Miles_Bin'] = (df['Miles'] > 500).astype(int)  # 0: 0-500, 1: 501-1000
    df['Receipts_Bin'] = (df['Receipts'] > 500).astype(int)  # 0: 0-500, 1: 501-1000
    
    # Create region identifier (0-7)
    df['Region'] = (df['Days_Bin'] * 4 + 
                   df['Miles_Bin'] * 2 + 
                   df['Receipts_Bin'])
    
    # Create region labels
    region_labels = {
        0: 'Low_Days_Low_Miles_Low_Receipts',
        1: 'Low_Days_Low_Miles_High_Receipts', 
        2: 'Low_Days_High_Miles_Low_Receipts',
        3: 'Low_Days_High_Miles_High_Receipts',
        4: 'High_Days_Low_Miles_Low_Receipts',
        5: 'High_Days_Low_Miles_High_Receipts',
        6: 'High_Days_High_Miles_Low_Receipts', 
        7: 'High_Days_High_Miles_High_Receipts'
    }
    
    df['Region_Label'] = df['Region'].map(region_labels)
    
    return df, region_labels


def analyze_region(df_region, region_id, region_label):
    """Analyze a specific region."""
    if len(df_region) < 5:
        return {
            'region_id': region_id,
            'region_label': region_label,
            'count': len(df_region),
            'error': 'Insufficient data (< 5 samples)'
        }
    
    # Basic statistics
    stats = {}
    for col in ['Days', 'Miles', 'Receipts', 'Reimb']:
        stats[col] = {
            'mean': float(df_region[col].mean()),
            'std': float(df_region[col].std()),
            'min': float(df_region[col].min()),
            'max': float(df_region[col].max()),
            'median': float(df_region[col].median())
        }
    
    # Formula accuracy test
    predicted = (df_region['Days'] * 100 + 
                df_region['Miles'] * 0.5 + 
                df_region['Receipts'])
    
    formula_r2 = r2_score(df_region['Reimb'], predicted)
    residuals = df_region['Reimb'] - predicted
    rmse = np.sqrt(mean_squared_error(df_region['Reimb'], predicted))
    
    # Linear regression within region
    X = df_region[['Days', 'Miles', 'Receipts']]
    y = df_region['Reimb']
    
    try:
        reg = LinearRegression()
        reg.fit(X, y)
        reg_pred = reg.predict(X)
        reg_r2 = r2_score(y, reg_pred)
        
        regression = {
            'r2_score': float(reg_r2),
            'coefficients': {
                'Days': float(reg.coef_[0]),
                'Miles': float(reg.coef_[1]), 
                'Receipts': float(reg.coef_[2])
            },
            'intercept': float(reg.intercept_)
        }
    except Exception as e:
        regression = {'error': f'Regression failed: {str(e)}'}
    
    return {
        'region_id': region_id,
        'region_label': region_label,
        'count': len(df_region),
        'percentage': len(df_region) / 1000 * 100,
        'statistics': stats,
        'formula_performance': {
            'r2_score': float(formula_r2),
            'rmse': float(rmse),
            'mean_residual': float(residuals.mean()),
            'std_residual': float(residuals.std())
        },
        'regression': regression
    }


def print_binned_results(region_results):
    """Print detailed results of binned analysis."""
    print("\n" + "="*80)
    print("BINNED ANALYSIS RESULTS - 8 PARAMETER REGIONS")
    print("="*80)
    
    print("\nRegion Definitions:")
    print("Days: Low (1-15) vs High (16-30)")
    print("Miles: Low (0-500) vs High (501-1000)")
    print("Receipts: Low (0-500) vs High (501-1000)")
    
    print(f"\n{'Region':>6} {'Label':>35} {'Count':>6} {'%':>5} {'Formula R²':>11} {'Reg R²':>8}")
    print("-" * 80)
    
    for result in region_results:
        if 'error' not in result:
            region_id = result['region_id']
            label_short = result['region_label'].replace('_', ' ')[:30]
            count = result['count']
            pct = result['percentage']
            formula_r2 = result['formula_performance']['r2_score']
            
            reg_r2 = 'N/A'
            if 'error' not in result['regression']:
                reg_r2 = f"{result['regression']['r2_score']:.3f}"
            
            print(f"R{region_id:>5} {label_short:>35} {count:>6} {pct:>5.1f} {formula_r2:>11.3f} {reg_r2:>8}")
        else:
            print(f"R{result['region_id']:>5} {'ERROR':>35} {result['count']:>6} {'N/A':>5} {'N/A':>11} {'N/A':>8}")


def create_binned_visualizations(df, region_results, output_dir):
    """Create visualizations for binned analysis."""
    plt.style.use('default')
    
    # Region distribution and performance
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Region counts
    region_counts = df['Region'].value_counts().sort_index()
    region_labels_short = [f'R{i}' for i in range(8)]
    ax1.bar(range(8), [region_counts.get(i, 0) for i in range(8)])
    ax1.set_xlabel('Region')
    ax1.set_ylabel('Count')
    ax1.set_title('Data Distribution Across 8 Regions')
    ax1.set_xticks(range(8))
    ax1.set_xticklabels(region_labels_short)
    
    # Formula R² by region
    formula_r2_values = []
    valid_regions = []
    for result in region_results:
        if 'error' not in result:
            formula_r2_values.append(result['formula_performance']['r2_score'])
            valid_regions.append(result['region_id'])
    
    if formula_r2_values:
        ax2.bar(valid_regions, formula_r2_values, alpha=0.7)
        ax2.set_xlabel('Region')
        ax2.set_ylabel('Formula R²')
        ax2.set_title('Formula Performance by Region')
        ax2.set_xticks(valid_regions)
        ax2.set_xticklabels([f'R{i}' for i in valid_regions])
    
    # Scatter plot colored by region
    colors = plt.cm.tab10(df['Region'])
    ax3.scatter(df['Days'], df['Reimb'], c=colors, alpha=0.6, s=20)
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Reimb')
    ax3.set_title('Days vs Reimb by Region')
    
    # Coefficient variation across regions
    coeff_data = []
    coeff_labels = []
    for result in region_results:
        if 'error' not in result and 'error' not in result['regression']:
            coeffs = result['regression']['coefficients']
            coeff_data.append([coeffs['Days'], coeffs['Miles'], coeffs['Receipts']])
            coeff_labels.append(f"R{result['region_id']}")
    
    if coeff_data:
        coeff_array = np.array(coeff_data)
        x_pos = np.arange(len(coeff_labels))
        width = 0.25
        
        ax4.bar(x_pos - width, coeff_array[:, 0], width, label='Days', alpha=0.7)
        ax4.bar(x_pos, coeff_array[:, 1], width, label='Miles', alpha=0.7)
        ax4.bar(x_pos + width, coeff_array[:, 2], width, label='Receipts', alpha=0.7)
        
        ax4.set_xlabel('Region')
        ax4.set_ylabel('Coefficient Value')
        ax4.set_title('Regression Coefficients by Region')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(coeff_labels)
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'binned_analysis.png', dpi=100, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Binned Analysis Across Parameter Half-Ranges')
    parser.add_argument('--csv', default='public.csv', help='Input CSV file (default: public.csv)')
    parser.add_argument('--output-dir', default='outputs/analysis_143208', 
                       help='Output directory')
    args = parser.parse_args()
    
    print("Binned Analysis Across Parameter Half-Ranges")
    print("=" * 50)
    
    # Load and bin data
    df, region_labels = load_and_bin_data(args.csv)
    print(f"Loaded {len(df)} records")
    print(f"Created 8 regions based on parameter half-ranges")
    
    # Analyze each region
    region_results = []
    for region_id in range(8):
        df_region = df[df['Region'] == region_id]
        region_label = region_labels[region_id]
        
        print(f"\nAnalyzing Region {region_id}: {region_label}")
        print(f"  Records: {len(df_region)}")
        
        result = analyze_region(df_region, region_id, region_label)
        region_results.append(result)
        
        if 'error' not in result:
            formula_r2 = result['formula_performance']['r2_score']
            print(f"  Formula R²: {formula_r2:.3f}")
        else:
            print(f"  Error: {result['error']}")
    
    # Create visualizations
    create_binned_visualizations(df, region_results, args.output_dir)
    print(f"\nVisualizations saved to: {args.output_dir}/")
    
    # Print detailed results
    print_binned_results(region_results)
    
    # Save results to JSON
    import json
    results_data = {
        'region_results': region_results,
        'region_labels': region_labels
    }
    
    output_file = Path(args.output_dir) / 'binned_analysis_results.json'
    with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return df, region_results


if __name__ == "__main__":
    main() 