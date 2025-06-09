#!/usr/bin/env python3
"""
Reimbursement Threshold Analysis: 2 Subsets (Above/Below $1000)

Analyzes reimbursement data by binning into 2 subsets based on reimbursement amount:
1. Below $1000 reimbursement
2. Above $1000 reimbursement

Applies linear formula fitting and polynomial analysis to each subset.

Usage:
    python src/analysis/binned/reimb_threshold_analysis.py --csv public.csv --threshold 1000
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ReimbThresholdAnalyzer:
    """Reimbursement threshold analysis for above/below threshold subsets."""
    
    def __init__(self, threshold=1000):
        """Initialize with reimbursement threshold."""
        self.threshold = threshold
        self.analysis_results = {
            'threshold': threshold,
            'linear_formulas': {},
            'polynomial_models': {},
            'subset_comparisons': {},
            'statistical_tests': {},
            'visualizations': []
        }
        
    def load_and_prepare_data(self, csv_file):
        """Load data and create threshold-based subsets."""
        print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # Convert to numeric
        for col in ['Days', 'Miles', 'Receipts', 'Reimb']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        # Create threshold-based subsets
        df['Reimb_Category'] = df['Reimb'].apply(lambda x: 'Above' if x >= self.threshold else 'Below')
        
        # Create interaction and ratio features
        df['Days_Miles'] = df['Days'] * df['Miles']
        df['Days_Receipts'] = df['Days'] * df['Receipts']
        df['Miles_Receipts'] = df['Miles'] * df['Receipts']
        df['Miles_per_Day'] = df['Miles'] / (df['Days'] + 1e-6)
        df['Receipts_per_Day'] = df['Receipts'] / (df['Days'] + 1e-6)
        df['Total_Expense'] = df['Miles'] + df['Receipts']
        df['Expense_Efficiency'] = df['Total_Expense'] / (df['Days'] + 1e-6)
        
        print(f"Data prepared: {len(df)} total records")
        
        # Show distribution by threshold
        below_count = len(df[df['Reimb_Category'] == 'Below'])
        above_count = len(df[df['Reimb_Category'] == 'Above'])
        
        print(f"\nThreshold distribution (${self.threshold}):")
        print(f"Below ${self.threshold}: {below_count:4d} records ({below_count/len(df)*100:.1f}%)")
        print(f"Above ${self.threshold}: {above_count:4d} records ({above_count/len(df)*100:.1f}%)")
        
        return df
    
    def fit_linear_formula(self, df_subset, subset_name):
        """Fit linear formula: A*Days + B*Miles + C*Receipts + D"""
        if len(df_subset) < 4:
            return {
                'subset': subset_name,
                'count': len(df_subset),
                'error': 'Insufficient data for linear fitting'
            }
        
        X = df_subset[['Days', 'Miles', 'Receipts']]
        y = df_subset['Reimb']
        
        # Fit linear regression
        reg = LinearRegression()
        reg.fit(X, y)
        
        # Predictions and metrics
        y_pred = reg.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Extract coefficients
        A, B, C = reg.coef_
        D = reg.intercept_
        
        # Calculate residuals
        residuals = y - y_pred
        
        # Calculate correlations
        correlations = {}
        for var1, var2 in [('Days', 'Miles'), ('Days', 'Receipts'), ('Miles', 'Receipts'),
                          ('Days', 'Reimb'), ('Miles', 'Reimb'), ('Receipts', 'Reimb')]:
            if len(df_subset) > 2:
                corr, p_val = pearsonr(df_subset[var1], df_subset[var2])
                correlations[f'{var1}_{var2}'] = {'correlation': float(corr), 'p_value': float(p_val)}
        
        return {
            'subset': subset_name,
            'count': len(df_subset),
            'formula': f"{A:.3f}*Days + {B:.3f}*Miles + {C:.3f}*Receipts + {D:.3f}",
            'coefficients': {
                'A_days': float(A),
                'B_miles': float(B), 
                'C_receipts': float(C),
                'D_intercept': float(D)
            },
            'performance': {
                'r2_score': float(r2),
                'rmse': float(rmse),
                'mean_residual': float(residuals.mean()),
                'std_residual': float(residuals.std()),
                'max_residual': float(residuals.abs().max())
            },
            'statistics': {
                'mean_days': float(df_subset['Days'].mean()),
                'mean_miles': float(df_subset['Miles'].mean()),
                'mean_receipts': float(df_subset['Receipts'].mean()),
                'mean_reimb': float(df_subset['Reimb'].mean()),
                'std_days': float(df_subset['Days'].std()),
                'std_miles': float(df_subset['Miles'].std()),
                'std_receipts': float(df_subset['Receipts'].std()),
                'std_reimb': float(df_subset['Reimb'].std()),
                'min_reimb': float(df_subset['Reimb'].min()),
                'max_reimb': float(df_subset['Reimb'].max())
            },
            'correlations': correlations
        }
    
    def fit_polynomial_models(self, df_subset, subset_name):
        """Fit polynomial models of various degrees."""
        if len(df_subset) < 10:
            return {'error': 'Insufficient data for polynomial fitting'}
        
        feature_sets = {
            'base': ['Days', 'Miles', 'Receipts'],
            'interactions': ['Days', 'Miles', 'Receipts', 'Days_Miles', 'Days_Receipts', 'Miles_Receipts'],
            'ratios': ['Days', 'Miles', 'Receipts', 'Miles_per_Day', 'Receipts_per_Day', 'Expense_Efficiency'],
            'combined': ['Days', 'Miles', 'Receipts', 'Days_Miles', 'Days_Receipts', 'Miles_Receipts', 
                        'Miles_per_Day', 'Receipts_per_Day', 'Expense_Efficiency']
        }
        
        models = {}
        
        for feature_name, features in feature_sets.items():
            # Check if all features exist
            available_features = [f for f in features if f in df_subset.columns]
            if len(available_features) < 3:
                continue
                
            X = df_subset[available_features]
            y = df_subset['Reimb']
            
            models[feature_name] = {}
            
            for degree in [1, 2, 3]:
                try:
                    poly_pipeline = Pipeline([
                        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                        ('linear', LinearRegression())
                    ])
                    
                    poly_pipeline.fit(X, y)
                    y_pred = poly_pipeline.predict(X)
                    r2 = r2_score(y, y_pred)
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    
                    models[feature_name][f'degree_{degree}'] = {
                        'r2_score': float(r2),
                        'rmse': float(rmse),
                        'features': available_features,
                        'n_features_generated': poly_pipeline.named_steps['poly'].n_output_features_
                    }
                    
                except Exception as e:
                    models[feature_name][f'degree_{degree}'] = {'error': str(e)}
        
        return models
    
    def analyze_threshold_subsets(self, df):
        """Perform comprehensive analysis on both threshold subsets."""
        print("\nPerforming threshold subset analysis...")
        
        # Analyze below threshold
        df_below = df[df['Reimb_Category'] == 'Below']
        print(f"Analyzing Below ${self.threshold}: {len(df_below)} records")
        
        if len(df_below) > 0:
            self.analysis_results['linear_formulas']['Below'] = self.fit_linear_formula(
                df_below, f"Below ${self.threshold}")
            self.analysis_results['polynomial_models']['Below'] = self.fit_polynomial_models(
                df_below, f"Below ${self.threshold}")
        
        # Analyze above threshold
        df_above = df[df['Reimb_Category'] == 'Above']
        print(f"Analyzing Above ${self.threshold}: {len(df_above)} records")
        
        if len(df_above) > 0:
            self.analysis_results['linear_formulas']['Above'] = self.fit_linear_formula(
                df_above, f"Above ${self.threshold}")
            self.analysis_results['polynomial_models']['Above'] = self.fit_polynomial_models(
                df_above, f"Above ${self.threshold}")
        
        # Analyze entire dataset for comparison
        print(f"Analyzing Entire Dataset: {len(df)} records")
        self.analysis_results['linear_formulas']['All'] = self.fit_linear_formula(
            df, "Entire Dataset")
        self.analysis_results['polynomial_models']['All'] = self.fit_polynomial_models(
            df, "Entire Dataset")
    
    def perform_statistical_tests(self, df):
        """Perform statistical tests between threshold subsets."""
        print("\nPerforming statistical tests...")
        
        df_below = df[df['Reimb_Category'] == 'Below']
        df_above = df[df['Reimb_Category'] == 'Above']
        
        if len(df_below) > 0 and len(df_above) > 0:
            tests = {}
            
            # T-tests for each variable
            for var in ['Days', 'Miles', 'Receipts']:
                t_stat, p_val = ttest_ind(df_below[var], df_above[var])
                tests[f'{var}_ttest'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_val),
                    'significant': p_val < 0.05,
                    'below_mean': float(df_below[var].mean()),
                    'above_mean': float(df_above[var].mean()),
                    'difference': float(df_above[var].mean() - df_below[var].mean())
                }
            
            # Correlation comparisons
            correlations_below = {}
            correlations_above = {}
            
            for var1, var2 in [('Days', 'Miles'), ('Days', 'Receipts'), ('Miles', 'Receipts')]:
                if len(df_below) > 2:
                    corr_below, _ = pearsonr(df_below[var1], df_below[var2])
                    correlations_below[f'{var1}_{var2}'] = float(corr_below)
                
                if len(df_above) > 2:
                    corr_above, _ = pearsonr(df_above[var1], df_above[var2])
                    correlations_above[f'{var1}_{var2}'] = float(corr_above)
            
            tests['correlation_comparison'] = {
                'below_threshold': correlations_below,
                'above_threshold': correlations_above
            }
            
            self.analysis_results['statistical_tests'] = tests
    
    def compare_threshold_subsets(self):
        """Compare and contrast patterns between threshold subsets."""
        print("\nComparing threshold subsets...")
        
        comparisons = {
            'coefficient_comparison': {},
            'performance_comparison': {},
            'subset_characteristics': {}
        }
        
        # Compare coefficients
        below_result = self.analysis_results['linear_formulas'].get('Below', {})
        above_result = self.analysis_results['linear_formulas'].get('Above', {})
        all_result = self.analysis_results['linear_formulas'].get('All', {})
        
        if ('error' not in below_result and 'error' not in above_result and 
            'coefficients' in below_result and 'coefficients' in above_result):
            
            for coef_name in ['A_days', 'B_miles', 'C_receipts', 'D_intercept']:
                below_val = below_result['coefficients'][coef_name]
                above_val = above_result['coefficients'][coef_name]
                
                comparisons['coefficient_comparison'][coef_name] = {
                    'below_threshold': below_val,
                    'above_threshold': above_val,
                    'difference': above_val - below_val,
                    'ratio': above_val / below_val if below_val != 0 else float('inf')
                }
        
        # Compare performance
        for subset_name in ['Below', 'Above', 'All']:
            result = self.analysis_results['linear_formulas'].get(subset_name, {})
            if 'error' not in result and 'performance' in result:
                comparisons['performance_comparison'][subset_name] = {
                    'r2_score': result['performance']['r2_score'],
                    'rmse': result['performance']['rmse'],
                    'count': result['count']
                }
        
        # Subset characteristics
        for subset_name in ['Below', 'Above']:
            result = self.analysis_results['linear_formulas'].get(subset_name, {})
            if 'error' not in result and 'statistics' in result:
                stats = result['statistics']
                comparisons['subset_characteristics'][subset_name] = {
                    'mean_days': stats['mean_days'],
                    'mean_miles': stats['mean_miles'],
                    'mean_receipts': stats['mean_receipts'],
                    'mean_reimb': stats['mean_reimb'],
                    'reimb_range': [stats['min_reimb'], stats['max_reimb']]
                }
        
        self.analysis_results['subset_comparisons'] = comparisons
    
    def create_threshold_visualizations(self, df, output_dir):
        """Create comprehensive threshold analysis visualizations."""
        print(f"\nCreating threshold visualizations in {output_dir}...")
        
        plt.style.use('default')
        
        # Create comprehensive dashboard
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Reimbursement distribution
        ax1 = plt.subplot(3, 4, 1)
        ax1.hist(df['Reimb'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(self.threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: ${self.threshold}')
        ax1.set_xlabel('Reimbursement ($)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Reimbursement Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Subset comparison - coefficient comparison
        ax2 = plt.subplot(3, 4, 2)
        below_result = self.analysis_results['linear_formulas'].get('Below', {})
        above_result = self.analysis_results['linear_formulas'].get('Above', {})
        
        if ('error' not in below_result and 'error' not in above_result and 
            'coefficients' in below_result and 'coefficients' in above_result):
            
            coef_names = ['A_days', 'B_miles', 'C_receipts']
            below_vals = [below_result['coefficients'][name] for name in coef_names]
            above_vals = [above_result['coefficients'][name] for name in coef_names]
            
            x = np.arange(len(coef_names))
            width = 0.35
            
            ax2.bar(x - width/2, below_vals, width, label=f'Below ${self.threshold}', alpha=0.8)
            ax2.bar(x + width/2, above_vals, width, label=f'Above ${self.threshold}', alpha=0.8)
            
            ax2.set_xlabel('Coefficient')
            ax2.set_ylabel('Value')
            ax2.set_title('Linear Formula Coefficients Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels(['A (Days)', 'B (Miles)', 'C (Receipts)'])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Performance comparison
        ax3 = plt.subplot(3, 4, 3)
        performance_data = []
        subset_names = []
        
        for subset in ['Below', 'Above', 'All']:
            result = self.analysis_results['linear_formulas'].get(subset, {})
            if 'error' not in result and 'performance' in result:
                performance_data.append(result['performance']['r2_score'])
                subset_names.append(subset)
        
        if performance_data:
            colors = ['orange', 'green', 'blue'][:len(performance_data)]
            bars = ax3.bar(subset_names, performance_data, color=colors, alpha=0.7)
            ax3.set_ylabel('R² Score')
            ax3.set_title('Linear Formula Performance Comparison')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, performance_data):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Scatter plot: Miles vs Receipts colored by threshold
        ax4 = plt.subplot(3, 4, 4)
        df_below = df[df['Reimb_Category'] == 'Below']
        df_above = df[df['Reimb_Category'] == 'Above']
        
        if len(df_below) > 0:
            ax4.scatter(df_below['Miles'], df_below['Receipts'], 
                       alpha=0.6, s=20, label=f'Below ${self.threshold}', color='orange')
        if len(df_above) > 0:
            ax4.scatter(df_above['Miles'], df_above['Receipts'], 
                       alpha=0.6, s=20, label=f'Above ${self.threshold}', color='green')
        
        ax4.set_xlabel('Miles')
        ax4.set_ylabel('Receipts')
        ax4.set_title('Miles vs Receipts by Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Box plots for each variable by threshold
        ax5 = plt.subplot(3, 4, 5)
        box_data = []
        box_labels = []
        
        for var in ['Days', 'Miles', 'Receipts']:
            if len(df_below) > 0:
                box_data.append(df_below[var])
                box_labels.append(f'{var}\nBelow')
            if len(df_above) > 0:
                box_data.append(df_above[var])
                box_labels.append(f'{var}\nAbove')
        
        if box_data:
            ax5.boxplot(box_data, labels=box_labels)
            ax5.set_title('Variable Distributions by Threshold')
            ax5.set_ylabel('Value')
            ax5.grid(True, alpha=0.3)
        
        # 6. Correlation heatmap - Below threshold
        ax6 = plt.subplot(3, 4, 6)
        if len(df_below) > 0:
            corr_below = df_below[['Days', 'Miles', 'Receipts', 'Reimb']].corr()
            sns.heatmap(corr_below, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax6, cbar_kws={'shrink': 0.8})
            ax6.set_title(f'Correlations: Below ${self.threshold}')
        
        # 7. Correlation heatmap - Above threshold
        ax7 = plt.subplot(3, 4, 7)
        if len(df_above) > 0:
            corr_above = df_above[['Days', 'Miles', 'Receipts', 'Reimb']].corr()
            sns.heatmap(corr_above, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax7, cbar_kws={'shrink': 0.8})
            ax7.set_title(f'Correlations: Above ${self.threshold}')
        
        # 8. Polynomial enhancement comparison
        ax8 = plt.subplot(3, 4, 8)
        linear_r2 = []
        poly_r2 = []
        enhancement_labels = []
        
        for subset in ['Below', 'Above']:
            linear_result = self.analysis_results['linear_formulas'].get(subset, {})
            poly_result = self.analysis_results['polynomial_models'].get(subset, {})
            
            if ('error' not in linear_result and 'performance' in linear_result and
                'error' not in poly_result):
                
                linear_r2.append(linear_result['performance']['r2_score'])
                
                # Find best polynomial R²
                best_poly_r2 = 0
                for feature_set, degrees in poly_result.items():
                    for degree, model_info in degrees.items():
                        if isinstance(model_info, dict) and 'r2_score' in model_info:
                            best_poly_r2 = max(best_poly_r2, model_info['r2_score'])
                
                poly_r2.append(best_poly_r2)
                enhancement_labels.append(subset)
        
        if linear_r2 and poly_r2:
            x = np.arange(len(enhancement_labels))
            width = 0.35
            
            ax8.bar(x - width/2, linear_r2, width, label='Linear', alpha=0.8, color='orange')
            ax8.bar(x + width/2, poly_r2, width, label='Polynomial', alpha=0.8, color='green')
            
            ax8.set_xlabel('Subset')
            ax8.set_ylabel('R² Score')
            ax8.set_title('Linear vs Polynomial Performance')
            ax8.set_xticks(x)
            ax8.set_xticklabels(enhancement_labels)
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # 9. Residual analysis
        ax9 = plt.subplot(3, 4, 9)
        all_residuals_below = []
        all_residuals_above = []
        
        for subset, color, label in [('Below', 'orange', f'Below ${self.threshold}'), 
                                   ('Above', 'green', f'Above ${self.threshold}')]:
            result = self.analysis_results['linear_formulas'].get(subset, {})
            if 'error' not in result and 'coefficients' in result:
                # Get subset data
                df_subset = df[df['Reimb_Category'] == subset]
                
                if len(df_subset) > 0:
                    X = df_subset[['Days', 'Miles', 'Receipts']]
                    y = df_subset['Reimb']
                    
                    # Recreate predictions
                    A = result['coefficients']['A_days']
                    B = result['coefficients']['B_miles']
                    C = result['coefficients']['C_receipts']
                    D = result['coefficients']['D_intercept']
                    
                    y_pred = A * X['Days'] + B * X['Miles'] + C * X['Receipts'] + D
                    residuals = y - y_pred
                    
                    ax9.scatter(y_pred, residuals, alpha=0.6, s=20, color=color, label=label)
        
        ax9.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax9.set_xlabel('Predicted Reimbursement')
        ax9.set_ylabel('Residuals')
        ax9.set_title('Residual Analysis by Threshold')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # 10. Sample size and performance
        ax10 = plt.subplot(3, 4, 10)
        sample_sizes = []
        r2_scores = []
        subset_labels = []
        
        for subset in ['Below', 'Above']:
            result = self.analysis_results['linear_formulas'].get(subset, {})
            if 'error' not in result and 'performance' in result:
                sample_sizes.append(result['count'])
                r2_scores.append(result['performance']['r2_score'])
                subset_labels.append(subset)
        
        if sample_sizes:
            colors = ['orange', 'green'][:len(sample_sizes)]
            scatter = ax10.scatter(sample_sizes, r2_scores, s=100, c=colors, alpha=0.7)
            
            for i, label in enumerate(subset_labels):
                ax10.annotate(label, (sample_sizes[i], r2_scores[i]), 
                             xytext=(5, 5), textcoords='offset points')
            
            ax10.set_xlabel('Sample Size')
            ax10.set_ylabel('R² Score')
            ax10.set_title('Performance vs Sample Size')
            ax10.grid(True, alpha=0.3)
        
        # 11. Statistical test results
        ax11 = plt.subplot(3, 4, 11)
        ax11.axis('off')
        
        if 'statistical_tests' in self.analysis_results:
            tests = self.analysis_results['statistical_tests']
            test_text = f"Statistical Tests (α=0.05):\n\n"
            
            for var in ['Days', 'Miles', 'Receipts']:
                test_key = f'{var}_ttest'
                if test_key in tests:
                    test_data = tests[test_key]
                    significance = "***" if test_data['significant'] else "n.s."
                    test_text += f"{var}:\n"
                    test_text += f"  Below: {test_data['below_mean']:.1f}\n"
                    test_text += f"  Above: {test_data['above_mean']:.1f}\n"
                    test_text += f"  p-value: {test_data['p_value']:.3f} {significance}\n\n"
            
            ax11.text(0.1, 0.9, test_text, transform=ax11.transAxes, 
                     fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # 12. Summary statistics
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        # Calculate summary statistics
        below_result = self.analysis_results['linear_formulas'].get('Below', {})
        above_result = self.analysis_results['linear_formulas'].get('Above', {})
        
        summary_text = f"""
        Threshold Analysis Summary:
        
        Threshold: ${self.threshold}
        Total Records: {len(df)}
        
        Below Threshold:
        • Count: {below_result.get('count', 0)}
        • R²: {below_result.get('performance', {}).get('r2_score', 0):.3f}
        
        Above Threshold:
        • Count: {above_result.get('count', 0)}
        • R²: {above_result.get('performance', {}).get('r2_score', 0):.3f}
        
        Performance Difference:
        • ΔR²: {(above_result.get('performance', {}).get('r2_score', 0) - below_result.get('performance', {}).get('r2_score', 0)):.3f}
        """
        
        ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'dashboard.png',
                   dpi=100, bbox_inches='tight')
        plt.close()
        
        self.analysis_results['visualizations'].append('dashboard.png')
        
        return str(Path(output_dir) / 'dashboard.png')
    
    def print_threshold_results(self):
        """Print comprehensive threshold analysis results."""
        print("\n" + "="*100)
        print(f"REIMBURSEMENT THRESHOLD ANALYSIS (${self.threshold})")
        print("="*100)
        
        # Linear formula results
        print(f"\n{'Subset':>15} {'Count':>6} {'A (Days)':>10} {'B (Miles)':>12} {'C (Receipts)':>14} {'D (Intercept)':>14} {'R²':>8} {'RMSE':>10}")
        print("-" * 100)
        
        for subset in ['Below', 'Above', 'All']:
            result = self.analysis_results['linear_formulas'].get(subset, {})
            if 'error' not in result:
                count = result['count']
                A = result['coefficients']['A_days']
                B = result['coefficients']['B_miles']
                C = result['coefficients']['C_receipts']
                D = result['coefficients']['D_intercept']
                r2 = result['performance']['r2_score']
                rmse = result['performance']['rmse']
                
                subset_label = f"{subset} ${self.threshold}" if subset in ['Below', 'Above'] else subset
                print(f"{subset_label:>15} {count:>6} {A:>10.3f} {B:>12.3f} {C:>14.3f} {D:>14.1f} {r2:>8.3f} {rmse:>10.1f}")
            else:
                print(f"{subset:>15} {'ERROR':>6} {'N/A':>10} {'N/A':>12} {'N/A':>14} {'N/A':>14} {'N/A':>8} {'N/A':>10}")
        
        # Statistical tests
        if 'statistical_tests' in self.analysis_results:
            print(f"\nStatistical Tests (t-tests):")
            print(f"{'Variable':>10} {'Below Mean':>12} {'Above Mean':>12} {'Difference':>12} {'p-value':>10} {'Significant':>12}")
            print("-" * 80)
            
            tests = self.analysis_results['statistical_tests']
            for var in ['Days', 'Miles', 'Receipts']:
                test_key = f'{var}_ttest'
                if test_key in tests:
                    test_data = tests[test_key]
                    significance = "Yes" if test_data['significant'] else "No"
                    print(f"{var:>10} {test_data['below_mean']:>12.1f} {test_data['above_mean']:>12.1f} "
                          f"{test_data['difference']:>12.1f} {test_data['p_value']:>10.3f} {significance:>12}")
        
        # Coefficient comparison
        if 'subset_comparisons' in self.analysis_results:
            comparisons = self.analysis_results['subset_comparisons']
            if 'coefficient_comparison' in comparisons:
                print(f"\nCoefficient Comparison:")
                print(f"{'Coefficient':>12} {'Below':>10} {'Above':>10} {'Difference':>12} {'Ratio':>10}")
                print("-" * 60)
                
                coef_comp = comparisons['coefficient_comparison']
                for coef_name in ['A_days', 'B_miles', 'C_receipts', 'D_intercept']:
                    if coef_name in coef_comp:
                        data = coef_comp[coef_name]
                        coef_display = coef_name.replace('_', ' ').title()
                        print(f"{coef_display:>12} {data['below_threshold']:>10.3f} {data['above_threshold']:>10.3f} "
                              f"{data['difference']:>12.3f} {data['ratio']:>10.2f}")
    
    def generate_threshold_report(self):
        """Generate comprehensive threshold analysis report."""
        report = {
            'analysis_overview': {
                'title': f'Reimbursement Threshold Analysis (${self.threshold})',
                'threshold': self.threshold,
                'timestamp': datetime.now().isoformat()
            },
            'linear_formula_analysis': self.analysis_results['linear_formulas'],
            'polynomial_model_analysis': self.analysis_results['polynomial_models'],
            'subset_comparisons': self.analysis_results['subset_comparisons'],
            'statistical_tests': self.analysis_results['statistical_tests'],
            'visualizations': self.analysis_results['visualizations']
        }
        
        return report


def main():
    parser = argparse.ArgumentParser(description='Reimbursement Threshold Analysis')
    parser.add_argument('--csv', default='public.csv', help='Input CSV file (default: public.csv)')
    parser.add_argument('--output-dir', default=None, 
                       help='Output directory (auto-generated if not specified)')
    parser.add_argument('--threshold', type=float, default=1000,
                       help='Reimbursement threshold (default: 1000)')
    parser.add_argument('--legacy', action='store_true',
                       help='Use legacy timestamped directory format')
    
    args = parser.parse_args()
    
    print(f"Reimbursement Threshold Analysis (${args.threshold})")
    print("=" * 60)
    
    # Create output directory with standardized naming
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        from core.utils import create_organized_output_dir
        output_dir = create_organized_output_dir(
            analysis_type="reimb_threshold",
            legacy=args.legacy,
            threshold=int(args.threshold)
        )
    
    print(f"Output directory: {output_dir}")
    
    # Initialize analyzer
    analyzer = ReimbThresholdAnalyzer(args.threshold)
    
    # Load and prepare data
    df = analyzer.load_and_prepare_data(args.csv)
    
    # Perform threshold analysis
    analyzer.analyze_threshold_subsets(df)
    
    # Perform statistical tests
    analyzer.perform_statistical_tests(df)
    
    # Compare subsets
    analyzer.compare_threshold_subsets()
    
    # Create visualizations
    viz_file = analyzer.create_threshold_visualizations(df, output_dir)
    print(f"Threshold visualization saved to: {viz_file}")
    
    # Print results
    analyzer.print_threshold_results()
    
    # Generate and save report with standardized naming
    report = analyzer.generate_threshold_report()
    
    # Use standardized file saving
    from core.utils import save_standardized_results
    saved_files = save_standardized_results(report, output_dir, "reimb_threshold")
    
    print(f"\nThreshold analysis results saved to: {saved_files['results']}")
    if 'summary' in saved_files:
        print(f"Summary saved to: {saved_files['summary']}")
    
    return analyzer, df, report


if __name__ == "__main__":
    main() 