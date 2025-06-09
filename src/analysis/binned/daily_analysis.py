#!/usr/bin/env python3
"""
Daily Binned Analysis: 14 Subsets (Days 1-14)

Analyzes reimbursement data by binning into 14 daily subsets and applying:
1. Linear formula fitting: A*Days + B*Miles + C*Receipts + D for each day
2. Polynomial feature analysis
3. Day-by-day pattern discovery
4. Performance comparison across days

Usage:
    python src/analysis/binned/daily_analysis.py --csv public.csv
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DailyBinnedAnalyzer:
    """Daily binned analysis for days 1-14."""
    
    def __init__(self, max_days=14):
        """Initialize with maximum days to analyze."""
        self.max_days = max_days
        self.analysis_results = {
            'linear_formulas': {},
            'polynomial_models': {},
            'daily_patterns': {},
            'comparisons': {},
            'visualizations': []
        }
        
    def load_and_prepare_data(self, csv_file):
        """Load data and filter for days 1-14."""
        print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # Convert to numeric
        for col in ['Days', 'Miles', 'Receipts', 'Reimb']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        # Filter for days 1-14
        df_filtered = df[(df['Days'] >= 1) & (df['Days'] <= self.max_days)].copy()
        
        # Create interaction and ratio features
        df_filtered['Days_Miles'] = df_filtered['Days'] * df_filtered['Miles']
        df_filtered['Days_Receipts'] = df_filtered['Days'] * df_filtered['Receipts']
        df_filtered['Miles_Receipts'] = df_filtered['Miles'] * df_filtered['Receipts']
        df_filtered['Miles_per_Day'] = df_filtered['Miles'] / (df_filtered['Days'] + 1e-6)
        df_filtered['Receipts_per_Day'] = df_filtered['Receipts'] / (df_filtered['Days'] + 1e-6)
        df_filtered['Total_Expense'] = df_filtered['Miles'] + df_filtered['Receipts']
        df_filtered['Expense_Efficiency'] = df_filtered['Total_Expense'] / (df_filtered['Days'] + 1e-6)
        
        print(f"Data prepared: {len(df_filtered)} records for days 1-{self.max_days}")
        print(f"Original dataset: {len(df)} records")
        
        # Show distribution by day
        day_counts = df_filtered['Days'].value_counts().sort_index()
        print(f"\nDaily distribution:")
        for day in range(1, self.max_days + 1):
            count = day_counts.get(day, 0)
            print(f"Day {day:2d}: {count:3d} records")
        
        return df_filtered
    
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
                'std_miles': float(df_subset['Miles'].std()),
                'std_receipts': float(df_subset['Receipts'].std()),
                'std_reimb': float(df_subset['Reimb'].std())
            },
            'correlations': correlations
        }
    
    def fit_polynomial_models(self, df_subset, subset_name):
        """Fit polynomial models of various degrees."""
        if len(df_subset) < 6:
            return {'error': 'Insufficient data for polynomial fitting'}
        
        feature_sets = {
            'base': ['Days', 'Miles', 'Receipts'],
            'interactions': ['Days', 'Miles', 'Receipts', 'Days_Miles', 'Days_Receipts', 'Miles_Receipts'],
            'ratios': ['Days', 'Miles', 'Receipts', 'Miles_per_Day', 'Receipts_per_Day', 'Expense_Efficiency']
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
    
    def analyze_all_days(self, df):
        """Perform comprehensive analysis on each day 1-14."""
        print("\nPerforming daily analysis...")
        
        # Analyze each day
        for day in range(1, self.max_days + 1):
            df_day = df[df['Days'] == day]
            day_name = f"Day_{day}"
            
            print(f"Analyzing Day {day}: {len(df_day)} records")
            
            if len(df_day) > 0:
                # Linear formula fitting
                self.analysis_results['linear_formulas'][day_name] = self.fit_linear_formula(
                    df_day, f"Day {day}")
                
                # Polynomial model fitting
                self.analysis_results['polynomial_models'][day_name] = self.fit_polynomial_models(
                    df_day, f"Day {day}")
                
                # Daily pattern analysis
                self.analysis_results['daily_patterns'][day_name] = {
                    'day': day,
                    'count': len(df_day),
                    'mean_reimb': float(df_day['Reimb'].mean()),
                    'std_reimb': float(df_day['Reimb'].std()),
                    'mean_miles': float(df_day['Miles'].mean()),
                    'mean_receipts': float(df_day['Receipts'].mean()),
                    'reimb_per_mile': float(df_day['Reimb'].sum() / df_day['Miles'].sum()) if df_day['Miles'].sum() > 0 else 0,
                    'reimb_per_receipt': float(df_day['Reimb'].sum() / df_day['Receipts'].sum()) if df_day['Receipts'].sum() > 0 else 0
                }
            else:
                self.analysis_results['linear_formulas'][day_name] = {
                    'subset': f"Day {day}",
                    'count': 0,
                    'error': 'No data for this day'
                }
                self.analysis_results['polynomial_models'][day_name] = {'error': 'No data for this day'}
                self.analysis_results['daily_patterns'][day_name] = {
                    'day': day,
                    'count': 0,
                    'error': 'No data for this day'
                }
    
    def compare_daily_patterns(self):
        """Compare and contrast patterns across days."""
        print("\nComparing daily patterns...")
        
        # Linear formula comparison
        linear_comparison = {
            'coefficient_trends': {},
            'performance_by_day': [],
            'daily_statistics': {}
        }
        
        # Extract data for comparison
        days = []
        coefficients = {'A_days': [], 'B_miles': [], 'C_receipts': [], 'D_intercept': []}
        performance_data = []
        
        for day_name, result in self.analysis_results['linear_formulas'].items():
            if 'error' not in result and 'coefficients' in result:
                day_num = int(day_name.split('_')[1])
                days.append(day_num)
                
                for coef_name, coef_value in result['coefficients'].items():
                    coefficients[coef_name].append(coef_value)
                
                performance_data.append({
                    'day': day_num,
                    'r2_score': result['performance']['r2_score'],
                    'rmse': result['performance']['rmse'],
                    'count': result['count']
                })
        
        # Calculate trends
        for coef_name, values in coefficients.items():
            if len(values) > 1:
                # Calculate correlation with day number
                if len(days) == len(values):
                    corr, p_val = pearsonr(days, values)
                    linear_comparison['coefficient_trends'][coef_name] = {
                        'correlation_with_day': float(corr),
                        'p_value': float(p_val),
                        'trend': 'increasing' if corr > 0.1 else 'decreasing' if corr < -0.1 else 'stable'
                    }
        
        linear_comparison['performance_by_day'] = sorted(performance_data, key=lambda x: x['day'])
        
        self.analysis_results['comparisons'] = {
            'linear_comparison': linear_comparison
        }
    
    def create_daily_visualizations(self, df, output_dir):
        """Create comprehensive daily analysis visualizations."""
        print(f"\nCreating daily visualizations in {output_dir}...")
        
        plt.style.use('default')
        
        # Create comprehensive dashboard
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Daily data distribution
        ax1 = plt.subplot(3, 4, 1)
        day_counts = df['Days'].value_counts().sort_index()
        days_range = range(1, self.max_days + 1)
        counts = [day_counts.get(day, 0) for day in days_range]
        
        bars = ax1.bar(days_range, counts, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Count')
        ax1.set_title('Data Distribution by Day (1-14)')
        ax1.set_xticks(days_range)
        
        # Add count labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # 2. Linear formula R² by day
        ax2 = plt.subplot(3, 4, 2)
        r2_scores = []
        valid_days = []
        
        for day in days_range:
            day_name = f"Day_{day}"
            if day_name in self.analysis_results['linear_formulas']:
                result = self.analysis_results['linear_formulas'][day_name]
                if 'error' not in result and 'performance' in result:
                    r2_scores.append(result['performance']['r2_score'])
                    valid_days.append(day)
        
        if r2_scores:
            ax2.plot(valid_days, r2_scores, 'o-', linewidth=2, markersize=6, color='orange')
            ax2.set_xlabel('Day')
            ax2.set_ylabel('R² Score')
            ax2.set_title('Linear Formula Performance by Day')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(range(1, self.max_days + 1))
        
        # 3. Coefficient trends
        ax3 = plt.subplot(3, 4, 3)
        coef_data = {'A_days': [], 'B_miles': [], 'C_receipts': []}
        coef_days = []
        
        for day in days_range:
            day_name = f"Day_{day}"
            if day_name in self.analysis_results['linear_formulas']:
                result = self.analysis_results['linear_formulas'][day_name]
                if 'error' not in result and 'coefficients' in result:
                    coef_days.append(day)
                    coef_data['A_days'].append(result['coefficients']['A_days'])
                    coef_data['B_miles'].append(result['coefficients']['B_miles'])
                    coef_data['C_receipts'].append(result['coefficients']['C_receipts'])
        
        if coef_days:
            ax3.plot(coef_days, coef_data['A_days'], 'o-', label='A (Days)', linewidth=2)
            ax3.plot(coef_days, coef_data['B_miles'], 's-', label='B (Miles)', linewidth=2)
            ax3.plot(coef_days, coef_data['C_receipts'], '^-', label='C (Receipts)', linewidth=2)
            ax3.set_xlabel('Day')
            ax3.set_ylabel('Coefficient Value')
            ax3.set_title('Coefficient Trends by Day')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xticks(range(1, self.max_days + 1))
        
        # 4. Mean reimbursement by day
        ax4 = plt.subplot(3, 4, 4)
        mean_reimb = []
        std_reimb = []
        reimb_days = []
        
        for day in days_range:
            df_day = df[df['Days'] == day]
            if len(df_day) > 0:
                reimb_days.append(day)
                mean_reimb.append(df_day['Reimb'].mean())
                std_reimb.append(df_day['Reimb'].std())
        
        if mean_reimb:
            ax4.errorbar(reimb_days, mean_reimb, yerr=std_reimb, 
                        marker='o', capsize=5, linewidth=2, markersize=6, color='green')
            ax4.set_xlabel('Day')
            ax4.set_ylabel('Mean Reimbursement')
            ax4.set_title('Mean Reimbursement by Day')
            ax4.grid(True, alpha=0.3)
            ax4.set_xticks(range(1, self.max_days + 1))
        
        # 5. Sample size vs performance
        ax5 = plt.subplot(3, 4, 5)
        sample_sizes = []
        performance_scores = []
        
        for day_name, result in self.analysis_results['linear_formulas'].items():
            if 'error' not in result and 'performance' in result:
                sample_sizes.append(result['count'])
                performance_scores.append(result['performance']['r2_score'])
        
        if sample_sizes:
            ax5.scatter(sample_sizes, performance_scores, alpha=0.7, s=50, color='purple')
            ax5.set_xlabel('Sample Size')
            ax5.set_ylabel('R² Score')
            ax5.set_title('Performance vs Sample Size')
            ax5.grid(True, alpha=0.3)
        
        # 6. Miles vs Receipts by day (colored)
        ax6 = plt.subplot(3, 4, 6)
        scatter = ax6.scatter(df['Miles'], df['Receipts'], c=df['Days'], 
                             cmap='viridis', alpha=0.6, s=20)
        ax6.set_xlabel('Miles')
        ax6.set_ylabel('Receipts')
        ax6.set_title('Miles vs Receipts (colored by Day)')
        plt.colorbar(scatter, ax=ax6, label='Day')
        
        # 7. Polynomial enhancement comparison
        ax7 = plt.subplot(3, 4, 7)
        linear_r2 = []
        poly_r2 = []
        enhancement_days = []
        
        for day in days_range:
            day_name = f"Day_{day}"
            linear_result = self.analysis_results['linear_formulas'].get(day_name, {})
            poly_result = self.analysis_results['polynomial_models'].get(day_name, {})
            
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
                enhancement_days.append(day)
        
        if linear_r2 and poly_r2:
            x = np.arange(len(enhancement_days))
            width = 0.35
            
            ax7.bar(x - width/2, linear_r2, width, label='Linear', alpha=0.8, color='orange')
            ax7.bar(x + width/2, poly_r2, width, label='Polynomial', alpha=0.8, color='green')
            
            ax7.set_xlabel('Day')
            ax7.set_ylabel('R² Score')
            ax7.set_title('Linear vs Polynomial Performance')
            ax7.set_xticks(x)
            ax7.set_xticklabels([f'D{d}' for d in enhancement_days])
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. Coefficient distribution boxplots
        ax8 = plt.subplot(3, 4, 8)
        if coef_data['A_days']:
            coef_matrix = np.array([coef_data['A_days'], coef_data['B_miles'], coef_data['C_receipts']])
            ax8.boxplot([coef_matrix[0], coef_matrix[1], coef_matrix[2]], 
                       labels=['A (Days)', 'B (Miles)', 'C (Receipts)'])
            ax8.set_title('Coefficient Distribution Across Days')
            ax8.set_ylabel('Coefficient Value')
            ax8.grid(True, alpha=0.3)
        
        # 9. Daily expense efficiency
        ax9 = plt.subplot(3, 4, 9)
        efficiency_data = []
        efficiency_days = []
        
        for day in days_range:
            df_day = df[df['Days'] == day]
            if len(df_day) > 0:
                efficiency = df_day['Total_Expense'].mean() / day
                efficiency_data.append(efficiency)
                efficiency_days.append(day)
        
        if efficiency_data:
            ax9.plot(efficiency_days, efficiency_data, 'o-', linewidth=2, markersize=6, color='red')
            ax9.set_xlabel('Day')
            ax9.set_ylabel('Expense Efficiency ($/day)')
            ax9.set_title('Daily Expense Efficiency')
            ax9.grid(True, alpha=0.3)
            ax9.set_xticks(range(1, self.max_days + 1))
        
        # 10. RMSE by day
        ax10 = plt.subplot(3, 4, 10)
        rmse_scores = []
        rmse_days = []
        
        for day in days_range:
            day_name = f"Day_{day}"
            if day_name in self.analysis_results['linear_formulas']:
                result = self.analysis_results['linear_formulas'][day_name]
                if 'error' not in result and 'performance' in result:
                    rmse_scores.append(result['performance']['rmse'])
                    rmse_days.append(day)
        
        if rmse_scores:
            ax10.plot(rmse_days, rmse_scores, 'o-', linewidth=2, markersize=6, color='brown')
            ax10.set_xlabel('Day')
            ax10.set_ylabel('RMSE')
            ax10.set_title('Prediction Error by Day')
            ax10.grid(True, alpha=0.3)
            ax10.set_xticks(range(1, self.max_days + 1))
        
        # 11. Correlation heatmap
        ax11 = plt.subplot(3, 4, 11)
        if len(df) > 0:
            corr_matrix = df[['Days', 'Miles', 'Receipts', 'Reimb']].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax11, cbar_kws={'shrink': 0.8})
            ax11.set_title('Overall Correlation Matrix')
        
        # 12. Summary statistics
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        # Calculate summary statistics
        valid_results = [r for r in self.analysis_results['linear_formulas'].values() 
                        if 'error' not in r and 'performance' in r]
        
        if valid_results:
            r2_scores_all = [r['performance']['r2_score'] for r in valid_results]
            mean_r2 = np.mean(r2_scores_all)
            std_r2 = np.std(r2_scores_all)
            
            summary_text = f"""
            Daily Analysis Summary:
            
            Days Analyzed: 1-{self.max_days}
            Valid Days: {len(valid_results)}
            Total Records: {len(df)}
            
            Linear Formula Performance:
            • Mean R²: {mean_r2:.3f} ± {std_r2:.3f}
            • Best Day R²: {max(r2_scores_all):.3f}
            • Worst Day R²: {min(r2_scores_all):.3f}
            
            Data Distribution:
            • Records per day: {len(df)/self.max_days:.1f} avg
            • Min day count: {min(counts)}
            • Max day count: {max(counts)}
            """
            
            ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, 
                     fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'daily_analysis_dashboard.png', 
                   dpi=100, bbox_inches='tight')
        plt.close()
        
        self.analysis_results['visualizations'].append('daily_analysis_dashboard.png')
        
        return str(Path(output_dir) / 'daily_analysis_dashboard.png')
    
    def print_daily_results(self):
        """Print comprehensive daily analysis results."""
        print("\n" + "="*100)
        print("DAILY BINNED ANALYSIS RESULTS (Days 1-14)")
        print("="*100)
        
        # Linear formula results
        print(f"\n{'Day':>4} {'Count':>6} {'A (Days)':>10} {'B (Miles)':>12} {'C (Receipts)':>14} {'D (Intercept)':>14} {'R²':>8} {'RMSE':>10}")
        print("-" * 100)
        
        for day in range(1, self.max_days + 1):
            day_name = f"Day_{day}"
            if day_name in self.analysis_results['linear_formulas']:
                result = self.analysis_results['linear_formulas'][day_name]
                if 'error' not in result:
                    count = result['count']
                    A = result['coefficients']['A_days']
                    B = result['coefficients']['B_miles']
                    C = result['coefficients']['C_receipts']
                    D = result['coefficients']['D_intercept']
                    r2 = result['performance']['r2_score']
                    rmse = result['performance']['rmse']
                    
                    print(f"{day:>4} {count:>6} {A:>10.3f} {B:>12.3f} {C:>14.3f} {D:>14.1f} {r2:>8.3f} {rmse:>10.1f}")
                else:
                    print(f"{day:>4} {'0':>6} {'N/A':>10} {'N/A':>12} {'N/A':>14} {'N/A':>14} {'N/A':>8} {'N/A':>10}")
        
        # Performance ranking
        valid_results = [(day, result) for day in range(1, self.max_days + 1) 
                        for day_name, result in [(f"Day_{day}", self.analysis_results['linear_formulas'].get(f"Day_{day}", {}))]
                        if 'error' not in result and 'performance' in result]
        
        if valid_results:
            sorted_results = sorted(valid_results, key=lambda x: x[1]['performance']['r2_score'], reverse=True)
            
            print(f"\nDaily Performance Ranking:")
            print(f"{'Rank':>4} {'Day':>4} {'R² Score':>10} {'RMSE':>10} {'Sample Size':>12}")
            print("-" * 50)
            
            for i, (day, result) in enumerate(sorted_results, 1):
                r2 = result['performance']['r2_score']
                rmse = result['performance']['rmse']
                count = result['count']
                print(f"{i:>4} {day:>4} {r2:>10.3f} {rmse:>10.1f} {count:>12}")
    
    def generate_daily_report(self):
        """Generate comprehensive daily analysis report."""
        report = {
            'analysis_overview': {
                'title': 'Daily Binned Analysis (Days 1-14)',
                'max_days': self.max_days,
                'timestamp': datetime.now().isoformat()
            },
            'linear_formula_analysis': self.analysis_results['linear_formulas'],
            'polynomial_model_analysis': self.analysis_results['polynomial_models'],
            'daily_patterns': self.analysis_results['daily_patterns'],
            'comparative_analysis': self.analysis_results['comparisons'],
            'visualizations': self.analysis_results['visualizations']
        }
        
        return report


def main():
    parser = argparse.ArgumentParser(description='Daily Binned Analysis (Days 1-14)')
    parser.add_argument('--csv', default='public.csv', help='Input CSV file (default: public.csv)')
    parser.add_argument('--output-dir', default='outputs/daily_analysis', 
                       help='Output directory')
    parser.add_argument('--max-days', type=int, default=14,
                       help='Maximum days to analyze (default: 14)')
    
    args = parser.parse_args()
    
    print("Daily Binned Analysis (Days 1-14)")
    print("=" * 50)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = DailyBinnedAnalyzer(args.max_days)
    
    # Load and prepare data
    df = analyzer.load_and_prepare_data(args.csv)
    
    # Perform daily analysis
    analyzer.analyze_all_days(df)
    
    # Compare daily patterns
    analyzer.compare_daily_patterns()
    
    # Create visualizations
    viz_file = analyzer.create_daily_visualizations(df, args.output_dir)
    print(f"Daily visualization saved to: {viz_file}")
    
    # Print results
    analyzer.print_daily_results()
    
    # Generate and save report
    report = analyzer.generate_daily_report()
    
    output_file = Path(args.output_dir) / 'daily_analysis_results.json'
    with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDaily analysis results saved to: {output_file}")
    
    return analyzer, df, report


if __name__ == "__main__":
    main() 