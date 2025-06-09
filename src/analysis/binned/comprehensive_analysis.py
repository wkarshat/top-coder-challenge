#!/usr/bin/env python3
"""
Comprehensive Binned Analysis with Linear Formula Fitting

Performs complete analysis including:
1. Linear formula fitting: A*Days + B*Miles + C*Receipts + D for each region and entire dataset
2. Polynomial feature analysis with configurable thresholds
3. Comprehensive reporting and visualization
4. Model comparison and contrast analysis

Usage:
    python src/analysis/binned/comprehensive_analysis.py --csv public.csv
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


class ComprehensiveBinnedAnalyzer:
    """Comprehensive binned analysis with linear formula fitting and polynomial analysis."""
    
    def __init__(self, config=None):
        """Initialize with configurable bounds."""
        self.config = config or {
            'days_threshold': 7,
            'miles_threshold': 700,
            'receipts_threshold': 1300,
            'polynomial_degrees': [1, 2, 3],
            'include_interactions': True,
            'include_ratios': True
        }
        
        self.analysis_results = {
            'linear_formulas': {},
            'polynomial_models': {},
            'comparisons': {},
            'visualizations': []
        }
        
    def load_and_prepare_data(self, csv_file):
        """Load data and create all necessary features."""
        print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # Convert to numeric
        for col in ['Days', 'Miles', 'Receipts', 'Reimb']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        # Create configurable binary bins
        df['Days_Bin'] = (df['Days'] > self.config['days_threshold']).astype(int)
        df['Miles_Bin'] = (df['Miles'] > self.config['miles_threshold']).astype(int)
        df['Receipts_Bin'] = (df['Receipts'] > self.config['receipts_threshold']).astype(int)
        
        # Create region identifier (0-7)
        df['Region'] = (df['Days_Bin'] * 4 + 
                       df['Miles_Bin'] * 2 + 
                       df['Receipts_Bin'])
        
        # Create region labels
        region_labels = {}
        for i in range(8):
            days_range = f"{self.config['days_threshold']+1}-14" if (i // 4) else f"1-{self.config['days_threshold']}"
            miles_range = f"{self.config['miles_threshold']+1}-1400" if ((i // 2) % 2) else f"1-{self.config['miles_threshold']}"
            receipts_range = f"{self.config['receipts_threshold']+1}-2600" if (i % 2) else f"1-{self.config['receipts_threshold']}"
            region_labels[i] = f"Days_{days_range}_Miles_{miles_range}_Receipts_{receipts_range}"
        
        df['Region_Label'] = df['Region'].map(region_labels)
        
        # Create interaction and ratio features
        if self.config['include_interactions']:
            df['Days_Miles'] = df['Days'] * df['Miles']
            df['Days_Receipts'] = df['Days'] * df['Receipts']
            df['Miles_Receipts'] = df['Miles'] * df['Receipts']
            df['Days_Miles_Receipts'] = df['Days'] * df['Miles'] * df['Receipts']
            
        if self.config['include_ratios']:
            df['Miles_per_Day'] = df['Miles'] / (df['Days'] + 1e-6)
            df['Receipts_per_Day'] = df['Receipts'] / (df['Days'] + 1e-6)
            df['Miles_per_Receipt'] = df['Miles'] / (df['Receipts'] + 1e-6)
            df['Total_Expense'] = df['Miles'] + df['Receipts']
            df['Expense_Efficiency'] = df['Total_Expense'] / (df['Days'] + 1e-6)
        
        print(f"Data prepared: {len(df)} records across {df['Region'].nunique()} regions")
        return df, region_labels
    
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
                'mean_reimb': float(df_subset['Reimb'].mean())
            }
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
            
            for degree in self.config['polynomial_degrees']:
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
    
    def analyze_all_regions(self, df, region_labels):
        """Perform comprehensive analysis on all regions and entire dataset."""
        print("\nPerforming comprehensive analysis...")
        
        # Analyze entire dataset
        print("Analyzing entire dataset...")
        self.analysis_results['linear_formulas']['entire_dataset'] = self.fit_linear_formula(df, 'Entire Dataset')
        self.analysis_results['polynomial_models']['entire_dataset'] = self.fit_polynomial_models(df, 'Entire Dataset')
        
        # Analyze each region
        for region_id in range(8):
            df_region = df[df['Region'] == region_id]
            region_name = f"Region_{region_id}"
            region_label = region_labels[region_id]
            
            print(f"Analyzing {region_name}: {len(df_region)} records")
            
            if len(df_region) > 0:
                # Linear formula fitting
                self.analysis_results['linear_formulas'][region_name] = self.fit_linear_formula(
                    df_region, f"{region_name} ({region_label})")
                
                # Polynomial model fitting
                self.analysis_results['polynomial_models'][region_name] = self.fit_polynomial_models(
                    df_region, f"{region_name} ({region_label})")
            else:
                self.analysis_results['linear_formulas'][region_name] = {
                    'subset': region_name,
                    'count': 0,
                    'error': 'No data in region'
                }
                self.analysis_results['polynomial_models'][region_name] = {'error': 'No data in region'}
    
    def compare_and_contrast_models(self):
        """Compare and contrast all models across regions."""
        print("\nComparing and contrasting models...")
        
        # Linear formula comparison
        linear_comparison = {
            'coefficient_analysis': {},
            'performance_ranking': [],
            'regional_patterns': {}
        }
        
        # Extract coefficients for comparison
        coefficients = {'A_days': [], 'B_miles': [], 'C_receipts': [], 'D_intercept': []}
        performance_data = []
        
        for region_name, result in self.analysis_results['linear_formulas'].items():
            if 'error' not in result and 'coefficients' in result:
                for coef_name, coef_value in result['coefficients'].items():
                    coefficients[coef_name].append(coef_value)
                
                performance_data.append({
                    'region': region_name,
                    'r2_score': result['performance']['r2_score'],
                    'rmse': result['performance']['rmse'],
                    'count': result['count']
                })
        
        # Coefficient statistics
        for coef_name, values in coefficients.items():
            if values:
                linear_comparison['coefficient_analysis'][coef_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'range': float(np.max(values) - np.min(values))
                }
        
        # Performance ranking
        linear_comparison['performance_ranking'] = sorted(
            performance_data, key=lambda x: x['r2_score'], reverse=True)
        
        # Polynomial model comparison
        polynomial_comparison = {
            'best_models_by_region': {},
            'feature_set_effectiveness': {},
            'degree_effectiveness': {}
        }
        
        # Find best model for each region
        for region_name, models in self.analysis_results['polynomial_models'].items():
            if 'error' not in models:
                best_r2 = -np.inf
                best_config = None
                
                for feature_set, degrees in models.items():
                    for degree, model_info in degrees.items():
                        if isinstance(model_info, dict) and 'r2_score' in model_info:
                            if model_info['r2_score'] > best_r2:
                                best_r2 = model_info['r2_score']
                                best_config = f"{feature_set}_{degree}"
                
                if best_config:
                    polynomial_comparison['best_models_by_region'][region_name] = {
                        'config': best_config,
                        'r2_score': best_r2
                    }
        
        self.analysis_results['comparisons'] = {
            'linear_comparison': linear_comparison,
            'polynomial_comparison': polynomial_comparison
        }
    
    def create_comprehensive_visualizations(self, df, output_dir):
        """Create comprehensive visualizations for all analyses."""
        print(f"\nCreating comprehensive visualizations in {output_dir}...")
        
        plt.style.use('default')
        
        # 1. Linear Formula Analysis Dashboard
        fig = plt.figure(figsize=(20, 16))
        
        # Coefficient comparison across regions
        ax1 = plt.subplot(3, 4, 1)
        linear_results = [r for r in self.analysis_results['linear_formulas'].values() 
                         if 'error' not in r and 'coefficients' in r]
        
        if linear_results:
            regions = [r['subset'].split('(')[0].strip() for r in linear_results]
            A_values = [r['coefficients']['A_days'] for r in linear_results]
            B_values = [r['coefficients']['B_miles'] for r in linear_results]
            C_values = [r['coefficients']['C_receipts'] for r in linear_results]
            
            x = np.arange(len(regions))
            width = 0.25
            
            ax1.bar(x - width, A_values, width, label='A (Days)', alpha=0.8)
            ax1.bar(x, B_values, width, label='B (Miles)', alpha=0.8)
            ax1.bar(x + width, C_values, width, label='C (Receipts)', alpha=0.8)
            
            ax1.set_xlabel('Region')
            ax1.set_ylabel('Coefficient Value')
            ax1.set_title('Linear Formula Coefficients by Region')
            ax1.set_xticks(x)
            ax1.set_xticklabels([r.replace('Region_', 'R').replace('Entire Dataset', 'All') for r in regions], rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # R² comparison: Linear vs Best Polynomial
        ax2 = plt.subplot(3, 4, 2)
        linear_r2 = []
        poly_r2 = []
        region_names = []
        
        for region_name in self.analysis_results['linear_formulas'].keys():
            linear_result = self.analysis_results['linear_formulas'][region_name]
            poly_result = self.analysis_results['polynomial_models'].get(region_name, {})
            
            if 'error' not in linear_result and 'performance' in linear_result:
                linear_r2.append(linear_result['performance']['r2_score'])
                
                # Find best polynomial R²
                best_poly_r2 = 0
                if 'error' not in poly_result:
                    for feature_set, degrees in poly_result.items():
                        for degree, model_info in degrees.items():
                            if isinstance(model_info, dict) and 'r2_score' in model_info:
                                best_poly_r2 = max(best_poly_r2, model_info['r2_score'])
                
                poly_r2.append(best_poly_r2)
                region_names.append(region_name.replace('Region_', 'R').replace('entire_dataset', 'All'))
        
        if linear_r2 and poly_r2:
            x = np.arange(len(region_names))
            width = 0.35
            
            ax2.bar(x - width/2, linear_r2, width, label='Linear Formula', alpha=0.8, color='orange')
            ax2.bar(x + width/2, poly_r2, width, label='Best Polynomial', alpha=0.8, color='green')
            
            ax2.set_xlabel('Region')
            ax2.set_ylabel('R² Score')
            ax2.set_title('Model Performance Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels(region_names, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Coefficient distribution
        ax3 = plt.subplot(3, 4, 3)
        if linear_results:
            coef_data = np.array([[r['coefficients']['A_days'], r['coefficients']['B_miles'], 
                                 r['coefficients']['C_receipts']] for r in linear_results])
            
            ax3.boxplot([coef_data[:, 0], coef_data[:, 1], coef_data[:, 2]], 
                       labels=['A (Days)', 'B (Miles)', 'C (Receipts)'])
            ax3.set_title('Coefficient Distribution Across Regions')
            ax3.set_ylabel('Coefficient Value')
            ax3.grid(True, alpha=0.3)
        
        # Regional data distribution
        ax4 = plt.subplot(3, 4, 4)
        region_counts = df['Region'].value_counts().sort_index()
        bars = ax4.bar(range(8), [region_counts.get(i, 0) for i in range(8)])
        ax4.set_xlabel('Region')
        ax4.set_ylabel('Count')
        ax4.set_title('Data Distribution by Region')
        ax4.set_xticks(range(8))
        ax4.set_xticklabels([f'R{i}' for i in range(8)])
        
        # Add count labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{int(height)}', ha='center', va='bottom')
        
        # Residual analysis
        ax5 = plt.subplot(3, 4, 5)
        if linear_results:
            all_residuals = []
            all_predictions = []
            
            for region_name, result in self.analysis_results['linear_formulas'].items():
                if 'error' not in result and region_name != 'entire_dataset':
                    # Get region data
                    if region_name.startswith('Region_'):
                        region_id = int(region_name.split('_')[1])
                        df_region = df[df['Region'] == region_id]
                        
                        if len(df_region) > 0:
                            X = df_region[['Days', 'Miles', 'Receipts']]
                            y = df_region['Reimb']
                            
                            # Recreate predictions
                            A = result['coefficients']['A_days']
                            B = result['coefficients']['B_miles']
                            C = result['coefficients']['C_receipts']
                            D = result['coefficients']['D_intercept']
                            
                            y_pred = A * X['Days'] + B * X['Miles'] + C * X['Receipts'] + D
                            residuals = y - y_pred
                            
                            all_residuals.extend(residuals)
                            all_predictions.extend(y_pred)
            
            if all_residuals:
                ax5.scatter(all_predictions, all_residuals, alpha=0.6, s=20)
                ax5.axhline(y=0, color='red', linestyle='--', alpha=0.8)
                ax5.set_xlabel('Predicted Reimbursement')
                ax5.set_ylabel('Residuals')
                ax5.set_title('Residual Analysis (All Regions)')
                ax5.grid(True, alpha=0.3)
        
        # Feature importance heatmap
        ax6 = plt.subplot(3, 4, 6)
        if len(linear_results) > 1:
            coef_matrix = np.array([[abs(r['coefficients']['A_days']), 
                                   abs(r['coefficients']['B_miles']), 
                                   abs(r['coefficients']['C_receipts'])] for r in linear_results[:-1]])  # Exclude entire dataset
            
            sns.heatmap(coef_matrix, 
                       xticklabels=['Days', 'Miles', 'Receipts'],
                       yticklabels=[f"R{i}" for i in range(len(coef_matrix))],
                       annot=True, fmt='.2f', cmap='viridis', ax=ax6)
            ax6.set_title('Absolute Coefficient Values by Region')
        
        # Performance vs sample size
        ax7 = plt.subplot(3, 4, 7)
        if linear_results:
            sample_sizes = [r['count'] for r in linear_results]
            r2_scores = [r['performance']['r2_score'] for r in linear_results]
            
            ax7.scatter(sample_sizes, r2_scores, alpha=0.7, s=50)
            ax7.set_xlabel('Sample Size')
            ax7.set_ylabel('R² Score')
            ax7.set_title('Performance vs Sample Size')
            ax7.grid(True, alpha=0.3)
            
            # Add region labels
            for i, (size, r2, result) in enumerate(zip(sample_sizes, r2_scores, linear_results)):
                region_label = result['subset'].split('(')[0].strip().replace('Region_', 'R').replace('Entire Dataset', 'All')
                ax7.annotate(region_label, (size, r2), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Polynomial degree effectiveness
        ax8 = plt.subplot(3, 4, 8)
        degree_performance = {1: [], 2: [], 3: []}
        
        for region_name, models in self.analysis_results['polynomial_models'].items():
            if 'error' not in models and region_name != 'entire_dataset':
                if 'base' in models:
                    for degree in [1, 2, 3]:
                        degree_key = f'degree_{degree}'
                        if degree_key in models['base'] and 'r2_score' in models['base'][degree_key]:
                            degree_performance[degree].append(models['base'][degree_key]['r2_score'])
        
        degrees = []
        mean_r2 = []
        std_r2 = []
        
        for degree, scores in degree_performance.items():
            if scores:
                degrees.append(degree)
                mean_r2.append(np.mean(scores))
                std_r2.append(np.std(scores))
        
        if degrees:
            ax8.errorbar(degrees, mean_r2, yerr=std_r2, marker='o', capsize=5, linewidth=2, markersize=8)
            ax8.set_xlabel('Polynomial Degree')
            ax8.set_ylabel('Mean R² Score')
            ax8.set_title('Polynomial Degree Effectiveness')
            ax8.grid(True, alpha=0.3)
            ax8.set_xticks(degrees)
        
        # 3D parameter space
        ax9 = plt.subplot(3, 4, 9, projection='3d')
        colors = plt.cm.tab10(df['Region'])
        scatter = ax9.scatter(df['Days'], df['Miles'], df['Receipts'], 
                             c=colors, alpha=0.6, s=20)
        ax9.set_xlabel('Days')
        ax9.set_ylabel('Miles')
        ax9.set_zlabel('Receipts')
        ax9.set_title('3D Parameter Space by Region')
        
        # Model complexity comparison
        ax10 = plt.subplot(3, 4, 10)
        complexity_data = []
        region_labels_plot = []
        
        for region_name, models in self.analysis_results['polynomial_models'].items():
            if 'error' not in models and region_name != 'entire_dataset':
                best_complexity = 1
                best_r2 = 0
                
                for feature_set, degrees in models.items():
                    for degree_key, model_info in degrees.items():
                        if isinstance(model_info, dict) and 'r2_score' in model_info:
                            if model_info['r2_score'] > best_r2:
                                best_r2 = model_info['r2_score']
                                if 'degree_1' in degree_key:
                                    best_complexity = 1
                                elif 'degree_2' in degree_key:
                                    best_complexity = 2
                                elif 'degree_3' in degree_key:
                                    best_complexity = 3
                
                complexity_data.append(best_complexity)
                region_labels_plot.append(region_name.replace('Region_', 'R'))
        
        if complexity_data:
            bars = ax10.bar(range(len(region_labels_plot)), complexity_data, alpha=0.7, color='purple')
            ax10.set_xlabel('Region')
            ax10.set_ylabel('Best Polynomial Degree')
            ax10.set_title('Optimal Model Complexity by Region')
            ax10.set_xticks(range(len(region_labels_plot)))
            ax10.set_xticklabels(region_labels_plot)
            ax10.set_ylim(0, 4)
        
        # Summary statistics
        ax11 = plt.subplot(3, 4, 11)
        ax11.axis('off')
        
        # Calculate summary statistics
        valid_linear_results = [r for r in self.analysis_results['linear_formulas'].values() 
                               if 'error' not in r and 'performance' in r]
        
        if valid_linear_results:
            linear_r2_scores = [r['performance']['r2_score'] for r in valid_linear_results]
            mean_linear_r2 = np.mean(linear_r2_scores)
            std_linear_r2 = np.std(linear_r2_scores)
            
            # Get polynomial summary
            poly_r2_scores = []
            for region_name, models in self.analysis_results['polynomial_models'].items():
                if 'error' not in models:
                    best_r2 = 0
                    for feature_set, degrees in models.items():
                        for degree, model_info in degrees.items():
                            if isinstance(model_info, dict) and 'r2_score' in model_info:
                                best_r2 = max(best_r2, model_info['r2_score'])
                    if best_r2 > 0:
                        poly_r2_scores.append(best_r2)
            
            mean_poly_r2 = np.mean(poly_r2_scores) if poly_r2_scores else 0
            
            summary_text = f"""
            Analysis Summary:
            
            Linear Formula Results:
            • Regions Analyzed: {len(valid_linear_results)}
            • Mean R²: {mean_linear_r2:.3f} ± {std_linear_r2:.3f}
            • Best Linear R²: {max(linear_r2_scores):.3f}
            
            Polynomial Results:
            • Mean Best R²: {mean_poly_r2:.3f}
            • Improvement: +{mean_poly_r2 - mean_linear_r2:.3f}
            
            Configuration:
            • Days threshold: {self.config['days_threshold']}
            • Miles threshold: {self.config['miles_threshold']}
            • Receipts threshold: {self.config['receipts_threshold']}
            """
            
            ax11.text(0.1, 0.9, summary_text, transform=ax11.transAxes, 
                     fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # Formula comparison table
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        if len(valid_linear_results) > 1:
            # Create table data
            table_data = []
            for result in valid_linear_results[:6]:  # Show first 6 regions
                region_name = result['subset'].split('(')[0].strip().replace('Region_', 'R').replace('Entire Dataset', 'All')
                A = result['coefficients']['A_days']
                B = result['coefficients']['B_miles']
                C = result['coefficients']['C_receipts']
                r2 = result['performance']['r2_score']
                table_data.append([region_name, f"{A:.2f}", f"{B:.3f}", f"{C:.3f}", f"{r2:.3f}"])
            
            table = ax12.table(cellText=table_data,
                              colLabels=['Region', 'A (Days)', 'B (Miles)', 'C (Receipts)', 'R²'],
                              cellLoc='center',
                              loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            ax12.set_title('Linear Formula Coefficients', pad=20)
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'comprehensive_analysis_dashboard.png', 
                   dpi=100, bbox_inches='tight')
        plt.close()
        
        self.analysis_results['visualizations'].append('comprehensive_analysis_dashboard.png')
        
        return str(Path(output_dir) / 'comprehensive_analysis_dashboard.png')
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report."""
        report = {
            'analysis_overview': {
                'title': 'Comprehensive Binned Analysis with Linear Formula Fitting',
                'configuration': self.config,
                'timestamp': datetime.now().isoformat()
            },
            'linear_formula_analysis': self.analysis_results['linear_formulas'],
            'polynomial_model_analysis': self.analysis_results['polynomial_models'],
            'comparative_analysis': self.analysis_results['comparisons'],
            'visualizations': self.analysis_results['visualizations']
        }
        
        return report
    
    def print_comprehensive_results(self):
        """Print comprehensive results with detailed analysis."""
        print("\n" + "="*120)
        print("COMPREHENSIVE BINNED ANALYSIS RESULTS")
        print("="*120)
        
        print(f"\nConfiguration:")
        print(f"Days threshold: {self.config['days_threshold']} (1-{self.config['days_threshold']} vs {self.config['days_threshold']+1}-14)")
        print(f"Miles threshold: {self.config['miles_threshold']} (1-{self.config['miles_threshold']} vs {self.config['miles_threshold']+1}-1400)")
        print(f"Receipts threshold: {self.config['receipts_threshold']} (1-{self.config['receipts_threshold']} vs {self.config['receipts_threshold']+1}-2600)")
        
        # Linear Formula Results
        print(f"\n{'='*60}")
        print("LINEAR FORMULA ANALYSIS: A*Days + B*Miles + C*Receipts + D")
        print(f"{'='*60}")
        
        print(f"\n{'Region':>15} {'Count':>6} {'A (Days)':>10} {'B (Miles)':>12} {'C (Receipts)':>14} {'D (Intercept)':>14} {'R²':>8} {'RMSE':>10}")
        print("-" * 120)
        
        for region_name, result in self.analysis_results['linear_formulas'].items():
            if 'error' not in result:
                region_display = region_name.replace('Region_', 'R').replace('entire_dataset', 'All Data')
                count = result['count']
                A = result['coefficients']['A_days']
                B = result['coefficients']['B_miles']
                C = result['coefficients']['C_receipts']
                D = result['coefficients']['D_intercept']
                r2 = result['performance']['r2_score']
                rmse = result['performance']['rmse']
                
                print(f"{region_display:>15} {count:>6} {A:>10.3f} {B:>12.3f} {C:>14.3f} {D:>14.1f} {r2:>8.3f} {rmse:>10.1f}")
            else:
                region_display = region_name.replace('Region_', 'R')
                print(f"{region_display:>15} {'ERROR':>6} {'N/A':>10} {'N/A':>12} {'N/A':>14} {'N/A':>14} {'N/A':>8} {'N/A':>10}")
        
        # Coefficient Analysis
        if 'linear_comparison' in self.analysis_results['comparisons']:
            coef_analysis = self.analysis_results['comparisons']['linear_comparison']['coefficient_analysis']
            
            print(f"\nCoefficient Statistics Across Regions:")
            print(f"{'Coefficient':>12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Range':>10}")
            print("-" * 70)
            
            for coef_name, stats in coef_analysis.items():
                display_name = coef_name.replace('_', ' ').title()
                print(f"{display_name:>12} {stats['mean']:>10.3f} {stats['std']:>10.3f} {stats['min']:>10.3f} {stats['max']:>10.3f} {stats['range']:>10.3f}")
        
        # Performance Ranking
        if 'linear_comparison' in self.analysis_results['comparisons']:
            ranking = self.analysis_results['comparisons']['linear_comparison']['performance_ranking']
            
            print(f"\nLinear Formula Performance Ranking:")
            print(f"{'Rank':>4} {'Region':>15} {'R² Score':>10} {'RMSE':>10} {'Sample Size':>12}")
            print("-" * 60)
            
            for i, entry in enumerate(ranking, 1):
                region_display = entry['region'].replace('Region_', 'R').replace('entire_dataset', 'All Data')
                print(f"{i:>4} {region_display:>15} {entry['r2_score']:>10.3f} {entry['rmse']:>10.1f} {entry['count']:>12}")
        
        # Polynomial Results Summary
        print(f"\n{'='*60}")
        print("POLYNOMIAL MODEL ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        print(f"\n{'Region':>15} {'Best Model':>25} {'R²':>8} {'Features':>10}")
        print("-" * 70)
        
        for region_name, models in self.analysis_results['polynomial_models'].items():
            region_display = region_name.replace('Region_', 'R').replace('entire_dataset', 'All Data')
            
            if 'error' not in models:
                best_r2 = -np.inf
                best_config = 'None'
                best_features = 0
                
                for feature_set, degrees in models.items():
                    for degree, model_info in degrees.items():
                        if isinstance(model_info, dict) and 'r2_score' in model_info:
                            if model_info['r2_score'] > best_r2:
                                best_r2 = model_info['r2_score']
                                best_config = f"{feature_set}_{degree}"
                                best_features = model_info.get('n_features_generated', 0)
                
                print(f"{region_display:>15} {best_config:>25} {best_r2:>8.3f} {best_features:>10}")
            else:
                print(f"{region_display:>15} {'ERROR':>25} {'N/A':>8} {'N/A':>10}")
        
        # Model Comparison Summary
        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        valid_linear = [r for r in self.analysis_results['linear_formulas'].values() 
                       if 'error' not in r and 'performance' in r]
        
        if valid_linear:
            linear_r2_scores = [r['performance']['r2_score'] for r in valid_linear]
            mean_linear_r2 = np.mean(linear_r2_scores)
            
            poly_r2_scores = []
            for region_name, models in self.analysis_results['polynomial_models'].items():
                if 'error' not in models:
                    best_r2 = 0
                    for feature_set, degrees in models.items():
                        for degree, model_info in degrees.items():
                            if isinstance(model_info, dict) and 'r2_score' in model_info:
                                best_r2 = max(best_r2, model_info['r2_score'])
                    if best_r2 > 0:
                        poly_r2_scores.append(best_r2)
            
            mean_poly_r2 = np.mean(poly_r2_scores) if poly_r2_scores else 0
            
            print(f"Linear Formula Average R²: {mean_linear_r2:.3f}")
            print(f"Polynomial Models Average R²: {mean_poly_r2:.3f}")
            print(f"Average Improvement: +{mean_poly_r2 - mean_linear_r2:.3f} R² points")
            print(f"Best Linear R²: {max(linear_r2_scores):.3f}")
            print(f"Best Polynomial R²: {max(poly_r2_scores) if poly_r2_scores else 0:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Binned Analysis with Linear Formula Fitting')
    parser.add_argument('--csv', default='public.csv', help='Input CSV file (default: public.csv)')
    parser.add_argument('--output-dir', default='outputs/analysis_143208', 
                       help='Output directory')
    parser.add_argument('--days-threshold', type=int, default=7,
                       help='Days threshold (default: 7)')
    parser.add_argument('--miles-threshold', type=int, default=700,
                       help='Miles threshold (default: 700)')
    parser.add_argument('--receipts-threshold', type=int, default=1300,
                       help='Receipts threshold (default: 1300)')
    parser.add_argument('--max-degree', type=int, default=3,
                       help='Maximum polynomial degree (default: 3)')
    
    args = parser.parse_args()
    
    # Configure analysis
    config = {
        'days_threshold': args.days_threshold,
        'miles_threshold': args.miles_threshold,
        'receipts_threshold': args.receipts_threshold,
        'polynomial_degrees': list(range(1, args.max_degree + 1)),
        'include_interactions': True,
        'include_ratios': True
    }
    
    print("Comprehensive Binned Analysis with Linear Formula Fitting")
    print("=" * 80)
    print(f"Configuration: {config}")
    
    # Initialize analyzer
    analyzer = ComprehensiveBinnedAnalyzer(config)
    
    # Load and prepare data
    df, region_labels = analyzer.load_and_prepare_data(args.csv)
    
    # Perform comprehensive analysis
    analyzer.analyze_all_regions(df, region_labels)
    
    # Compare and contrast models
    analyzer.compare_and_contrast_models()
    
    # Create visualizations
    viz_file = analyzer.create_comprehensive_visualizations(df, args.output_dir)
    print(f"Comprehensive visualization saved to: {viz_file}")
    
    # Print results
    analyzer.print_comprehensive_results()
    
    # Generate and save comprehensive report
    report = analyzer.generate_comprehensive_report()
    
    output_file = Path(args.output_dir) / 'comprehensive_analysis_results.json'
    with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nComprehensive analysis results saved to: {output_file}")
    
    return analyzer, df, report


if __name__ == "__main__":
    main() 