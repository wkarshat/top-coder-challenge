#!/usr/bin/env python3
"""
Daily Rate Analysis: 56 Subregions (14 Days × 4 Regions per Day)

Bins data into 56 subregions based on:
- Days: 1-14 individual days
- Miles per day: ≤100 vs >100 per day
- Receipts per day: ≤$100 vs >$100 per day

Analyzes linear formula fitting and looks for reimbursement thresholds.

Usage:
    python src/analysis/binned/daily_rate_analysis.py --csv public.csv
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
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DailyRateAnalyzer:
    """Daily rate analysis for 56 subregions."""
    
    def __init__(self, miles_per_day_threshold=100, receipts_per_day_threshold=100):
        """Initialize with daily rate thresholds."""
        self.miles_threshold = miles_per_day_threshold
        self.receipts_threshold = receipts_per_day_threshold
        self.max_days = 14
        self.analysis_results = {
            'configuration': {
                'miles_per_day_threshold': miles_per_day_threshold,
                'receipts_per_day_threshold': receipts_per_day_threshold,
                'max_days': self.max_days
            },
            'subregion_analysis': {},
            'threshold_analysis': {},
            'daily_patterns': {},
            'reimbursement_limits': {},
            'visualizations': []
        }
        
    def load_and_prepare_data(self, csv_file):
        """Load data and create daily rate features."""
        print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # Convert to numeric
        for col in ['Days', 'Miles', 'Receipts', 'Reimb']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        # Filter for days 1-14
        df_filtered = df[(df['Days'] >= 1) & (df['Days'] <= self.max_days)].copy()
        
        # Calculate daily rates
        df_filtered['Miles_per_Day'] = df_filtered['Miles'] / df_filtered['Days']
        df_filtered['Receipts_per_Day'] = df_filtered['Receipts'] / df_filtered['Days']
        df_filtered['Reimb_per_Day'] = df_filtered['Reimb'] / df_filtered['Days']
        
        # Create binary classifications for each day
        for day in range(1, self.max_days + 1):
            df_day = df_filtered[df_filtered['Days'] == day].copy()
            
            if len(df_day) > 0:
                # For this specific day, check if miles/receipts exceed daily thresholds
                miles_limit = self.miles_threshold * day
                receipts_limit = self.receipts_threshold * day
                
                df_day['Miles_Category'] = df_day['Miles'].apply(
                    lambda x: 'High' if x > miles_limit else 'Low')
                df_day['Receipts_Category'] = df_day['Receipts'].apply(
                    lambda x: 'High' if x > receipts_limit else 'Low')
                
                # Create region identifier (0-3)
                df_day['Region'] = (
                    (df_day['Miles_Category'] == 'High').astype(int) * 2 +
                    (df_day['Receipts_Category'] == 'High').astype(int)
                )
                
                # Create subregion identifier (day-1)*4 + region
                df_day['Subregion'] = (day - 1) * 4 + df_day['Region']
                
                # Store back to main dataframe
                if 'Subregion' not in df_filtered.columns:
                    df_filtered['Subregion'] = -1
                    df_filtered['Miles_Category'] = 'Unknown'
                    df_filtered['Receipts_Category'] = 'Unknown'
                    df_filtered['Region'] = -1
                
                df_filtered.loc[df_filtered['Days'] == day, 'Subregion'] = df_day['Subregion']
                df_filtered.loc[df_filtered['Days'] == day, 'Miles_Category'] = df_day['Miles_Category']
                df_filtered.loc[df_filtered['Days'] == day, 'Receipts_Category'] = df_day['Receipts_Category']
                df_filtered.loc[df_filtered['Days'] == day, 'Region'] = df_day['Region']
        
        print(f"Data prepared: {len(df_filtered)} records for days 1-{self.max_days}")
        
        # Show distribution by subregion
        subregion_counts = df_filtered['Subregion'].value_counts().sort_index()
        print(f"\nSubregion distribution (56 total possible):")
        populated_subregions = len(subregion_counts[subregion_counts > 0])
        print(f"Populated subregions: {populated_subregions}/56")
        
        return df_filtered
    
    def get_subregion_label(self, subregion_id):
        """Get descriptive label for subregion."""
        if subregion_id < 0:
            return "Unknown"
        
        day = (subregion_id // 4) + 1
        region = subregion_id % 4
        
        miles_limit = self.miles_threshold * day
        receipts_limit = self.receipts_threshold * day
        
        miles_desc = f"≤{miles_limit}" if region < 2 else f">{miles_limit}"
        receipts_desc = f"≤${receipts_limit}" if region % 2 == 0 else f">${receipts_limit}"
        
        return f"Day{day}_M{miles_desc}_R{receipts_desc}"
    
    def fit_linear_formula(self, df_subset, subregion_id):
        """Fit linear formula: A*Days + B*Miles + C*Receipts + D"""
        if len(df_subset) < 4:
            return {
                'subregion_id': subregion_id,
                'subregion_label': self.get_subregion_label(subregion_id),
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
        
        # Calculate daily rates and limits
        day = (subregion_id // 4) + 1
        mean_miles_per_day = df_subset['Miles_per_Day'].mean()
        mean_receipts_per_day = df_subset['Receipts_per_Day'].mean()
        mean_reimb_per_day = df_subset['Reimb_per_Day'].mean()
        
        return {
            'subregion_id': subregion_id,
            'subregion_label': self.get_subregion_label(subregion_id),
            'day': day,
            'region': subregion_id % 4,
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
            'daily_rates': {
                'mean_miles_per_day': float(mean_miles_per_day),
                'mean_receipts_per_day': float(mean_receipts_per_day),
                'mean_reimb_per_day': float(mean_reimb_per_day),
                'max_miles_per_day': float(df_subset['Miles_per_Day'].max()),
                'max_receipts_per_day': float(df_subset['Receipts_per_Day'].max()),
                'max_reimb_per_day': float(df_subset['Reimb_per_Day'].max())
            },
            'statistics': {
                'mean_days': float(df_subset['Days'].mean()),
                'mean_miles': float(df_subset['Miles'].mean()),
                'mean_receipts': float(df_subset['Receipts'].mean()),
                'mean_reimb': float(df_subset['Reimb'].mean()),
                'std_reimb': float(df_subset['Reimb'].std()),
                'min_reimb': float(df_subset['Reimb'].min()),
                'max_reimb': float(df_subset['Reimb'].max())
            }
        }
    
    def analyze_all_subregions(self, df):
        """Perform comprehensive analysis on all 56 subregions."""
        print("\nPerforming subregion analysis...")
        
        # Analyze each subregion
        for subregion_id in range(56):  # 14 days × 4 regions
            df_subregion = df[df['Subregion'] == subregion_id]
            
            day = (subregion_id // 4) + 1
            region = subregion_id % 4
            label = self.get_subregion_label(subregion_id)
            
            if len(df_subregion) > 0:
                print(f"Analyzing Subregion {subregion_id:2d} ({label}): {len(df_subregion)} records")
                
                # Linear formula fitting
                result = self.fit_linear_formula(df_subregion, subregion_id)
                self.analysis_results['subregion_analysis'][subregion_id] = result
            else:
                print(f"Subregion {subregion_id:2d} ({label}): No data")
                self.analysis_results['subregion_analysis'][subregion_id] = {
                    'subregion_id': subregion_id,
                    'subregion_label': label,
                    'day': day,
                    'region': region,
                    'count': 0,
                    'error': 'No data in subregion'
                }
    
    def analyze_reimbursement_thresholds(self, df):
        """Analyze potential reimbursement thresholds and limits."""
        print("\nAnalyzing reimbursement thresholds...")
        
        threshold_analysis = {
            'daily_rate_limits': {},
            'reimbursement_caps': {},
            'policy_patterns': {}
        }
        
        # Analyze daily rate patterns
        for day in range(1, self.max_days + 1):
            df_day = df[df['Days'] == day]
            
            if len(df_day) > 0:
                # Calculate percentiles for daily rates
                miles_per_day = df_day['Miles_per_Day']
                receipts_per_day = df_day['Receipts_per_Day']
                reimb_per_day = df_day['Reimb_per_Day']
                
                threshold_analysis['daily_rate_limits'][day] = {
                    'miles_per_day': {
                        'mean': float(miles_per_day.mean()),
                        'median': float(miles_per_day.median()),
                        'p90': float(miles_per_day.quantile(0.9)),
                        'p95': float(miles_per_day.quantile(0.95)),
                        'max': float(miles_per_day.max()),
                        'exceeds_200': int((miles_per_day > 200).sum()),
                        'exceeds_300': int((miles_per_day > 300).sum())
                    },
                    'receipts_per_day': {
                        'mean': float(receipts_per_day.mean()),
                        'median': float(receipts_per_day.median()),
                        'p90': float(receipts_per_day.quantile(0.9)),
                        'p95': float(receipts_per_day.quantile(0.95)),
                        'max': float(receipts_per_day.max()),
                        'exceeds_200': int((receipts_per_day > 200).sum()),
                        'exceeds_300': int((receipts_per_day > 300).sum())
                    },
                    'reimb_per_day': {
                        'mean': float(reimb_per_day.mean()),
                        'median': float(reimb_per_day.median()),
                        'p90': float(reimb_per_day.quantile(0.9)),
                        'p95': float(reimb_per_day.quantile(0.95)),
                        'max': float(reimb_per_day.max())
                    }
                }
        
        # Look for reimbursement caps
        all_reimb_per_day = df['Reimb_per_Day']
        threshold_analysis['reimbursement_caps'] = {
            'overall_daily_reimb': {
                'mean': float(all_reimb_per_day.mean()),
                'median': float(all_reimb_per_day.median()),
                'p90': float(all_reimb_per_day.quantile(0.9)),
                'p95': float(all_reimb_per_day.quantile(0.95)),
                'p99': float(all_reimb_per_day.quantile(0.99)),
                'max': float(all_reimb_per_day.max())
            },
            'potential_caps': {
                'daily_200': int((all_reimb_per_day <= 200).sum()),
                'daily_300': int((all_reimb_per_day <= 300).sum()),
                'daily_400': int((all_reimb_per_day <= 400).sum()),
                'daily_500': int((all_reimb_per_day <= 500).sum())
            }
        }
        
        self.analysis_results['threshold_analysis'] = threshold_analysis
    
    def analyze_daily_patterns(self, df):
        """Analyze patterns across days and regions."""
        print("\nAnalyzing daily patterns...")
        
        patterns = {
            'by_day': {},
            'by_region': {},
            'coefficient_trends': {}
        }
        
        # Analyze by day
        for day in range(1, self.max_days + 1):
            day_results = []
            for region in range(4):
                subregion_id = (day - 1) * 4 + region
                result = self.analysis_results['subregion_analysis'].get(subregion_id, {})
                if 'error' not in result and 'performance' in result:
                    day_results.append(result)
            
            if day_results:
                patterns['by_day'][day] = {
                    'populated_regions': len(day_results),
                    'total_records': sum(r['count'] for r in day_results),
                    'avg_r2': float(np.mean([r['performance']['r2_score'] for r in day_results])),
                    'best_r2': float(max([r['performance']['r2_score'] for r in day_results])),
                    'avg_A_coef': float(np.mean([r['coefficients']['A_days'] for r in day_results])),
                    'avg_B_coef': float(np.mean([r['coefficients']['B_miles'] for r in day_results])),
                    'avg_C_coef': float(np.mean([r['coefficients']['C_receipts'] for r in day_results]))
                }
        
        # Analyze by region type
        for region in range(4):
            region_results = []
            for day in range(1, self.max_days + 1):
                subregion_id = (day - 1) * 4 + region
                result = self.analysis_results['subregion_analysis'].get(subregion_id, {})
                if 'error' not in result and 'performance' in result:
                    region_results.append(result)
            
            if region_results:
                patterns['by_region'][region] = {
                    'populated_days': len(region_results),
                    'total_records': sum(r['count'] for r in region_results),
                    'avg_r2': float(np.mean([r['performance']['r2_score'] for r in region_results])),
                    'best_r2': float(max([r['performance']['r2_score'] for r in region_results])),
                    'avg_A_coef': float(np.mean([r['coefficients']['A_days'] for r in region_results])),
                    'avg_B_coef': float(np.mean([r['coefficients']['B_miles'] for r in region_results])),
                    'avg_C_coef': float(np.mean([r['coefficients']['C_receipts'] for r in region_results]))
                }
        
        self.analysis_results['daily_patterns'] = patterns
    
    def create_comprehensive_visualizations(self, df, output_dir):
        """Create comprehensive visualizations for daily rate analysis."""
        print(f"\nCreating visualizations in {output_dir}...")
        
        plt.style.use('default')
        
        # Create comprehensive dashboard
        fig = plt.figure(figsize=(24, 20))
        
        # 1. Subregion data distribution heatmap
        ax1 = plt.subplot(4, 6, 1)
        
        # Create heatmap data
        heatmap_data = np.zeros((self.max_days, 4))
        for subregion_id, result in self.analysis_results['subregion_analysis'].items():
            if 'error' not in result:
                day = result['day'] - 1  # 0-indexed for array
                region = result['region']
                heatmap_data[day, region] = result['count']
        
        sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='Blues', ax=ax1,
                   xticklabels=['Low M, Low R', 'Low M, High R', 'High M, Low R', 'High M, High R'],
                   yticklabels=[f'Day {i+1}' for i in range(self.max_days)])
        ax1.set_title('Data Distribution by Subregion')
        ax1.set_xlabel('Region Type')
        ax1.set_ylabel('Day')
        
        # 2. R² performance heatmap
        ax2 = plt.subplot(4, 6, 2)
        
        r2_data = np.full((self.max_days, 4), np.nan)
        for subregion_id, result in self.analysis_results['subregion_analysis'].items():
            if 'error' not in result and 'performance' in result:
                day = result['day'] - 1
                region = result['region']
                r2_data[day, region] = result['performance']['r2_score']
        
        sns.heatmap(r2_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax2,
                   xticklabels=['Low M, Low R', 'Low M, High R', 'High M, Low R', 'High M, High R'],
                   yticklabels=[f'Day {i+1}' for i in range(self.max_days)])
        ax2.set_title('R² Performance by Subregion')
        ax2.set_xlabel('Region Type')
        ax2.set_ylabel('Day')
        
        # 3. A coefficient (Days) heatmap
        ax3 = plt.subplot(4, 6, 3)
        
        a_coef_data = np.full((self.max_days, 4), np.nan)
        for subregion_id, result in self.analysis_results['subregion_analysis'].items():
            if 'error' not in result and 'coefficients' in result:
                day = result['day'] - 1
                region = result['region']
                a_coef_data[day, region] = result['coefficients']['A_days']
        
        sns.heatmap(a_coef_data, annot=True, fmt='.1f', cmap='coolwarm', center=0, ax=ax3,
                   xticklabels=['Low M, Low R', 'Low M, High R', 'High M, Low R', 'High M, High R'],
                   yticklabels=[f'Day {i+1}' for i in range(self.max_days)])
        ax3.set_title('A Coefficient (Days) by Subregion')
        ax3.set_xlabel('Region Type')
        ax3.set_ylabel('Day')
        
        # 4. B coefficient (Miles) heatmap
        ax4 = plt.subplot(4, 6, 4)
        
        b_coef_data = np.full((self.max_days, 4), np.nan)
        for subregion_id, result in self.analysis_results['subregion_analysis'].items():
            if 'error' not in result and 'coefficients' in result:
                day = result['day'] - 1
                region = result['region']
                b_coef_data[day, region] = result['coefficients']['B_miles']
        
        sns.heatmap(b_coef_data, annot=True, fmt='.3f', cmap='coolwarm', center=0, ax=ax4,
                   xticklabels=['Low M, Low R', 'Low M, High R', 'High M, Low R', 'High M, High R'],
                   yticklabels=[f'Day {i+1}' for i in range(self.max_days)])
        ax4.set_title('B Coefficient (Miles) by Subregion')
        ax4.set_xlabel('Region Type')
        ax4.set_ylabel('Day')
        
        # 5. C coefficient (Receipts) heatmap
        ax5 = plt.subplot(4, 6, 5)
        
        c_coef_data = np.full((self.max_days, 4), np.nan)
        for subregion_id, result in self.analysis_results['subregion_analysis'].items():
            if 'error' not in result and 'coefficients' in result:
                day = result['day'] - 1
                region = result['region']
                c_coef_data[day, region] = result['coefficients']['C_receipts']
        
        sns.heatmap(c_coef_data, annot=True, fmt='.3f', cmap='coolwarm', center=0, ax=ax5,
                   xticklabels=['Low M, Low R', 'Low M, High R', 'High M, Low R', 'High M, High R'],
                   yticklabels=[f'Day {i+1}' for i in range(self.max_days)])
        ax5.set_title('C Coefficient (Receipts) by Subregion')
        ax5.set_xlabel('Region Type')
        ax5.set_ylabel('Day')
        
        # 6. Daily rate distribution
        ax6 = plt.subplot(4, 6, 6)
        
        if 'threshold_analysis' in self.analysis_results:
            days = []
            miles_p95 = []
            receipts_p95 = []
            
            for day, data in self.analysis_results['threshold_analysis']['daily_rate_limits'].items():
                days.append(day)
                miles_p95.append(data['miles_per_day']['p95'])
                receipts_p95.append(data['receipts_per_day']['p95'])
            
            ax6.plot(days, miles_p95, 'o-', label='Miles/Day (95th %ile)', linewidth=2)
            ax6.plot(days, receipts_p95, 's-', label='Receipts/Day (95th %ile)', linewidth=2)
            ax6.axhline(y=200, color='red', linestyle='--', alpha=0.7, label='$200 threshold')
            ax6.set_xlabel('Day')
            ax6.set_ylabel('Daily Rate')
            ax6.set_title('Daily Rate Patterns (95th Percentile)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7-12. Performance by day and region
        for i, region in enumerate(range(4)):
            ax = plt.subplot(4, 6, 7 + i)
            
            days = []
            r2_scores = []
            
            for day in range(1, self.max_days + 1):
                subregion_id = (day - 1) * 4 + region
                result = self.analysis_results['subregion_analysis'].get(subregion_id, {})
                if 'error' not in result and 'performance' in result:
                    days.append(day)
                    r2_scores.append(result['performance']['r2_score'])
            
            if days:
                ax.plot(days, r2_scores, 'o-', linewidth=2, markersize=6)
                ax.set_xlabel('Day')
                ax.set_ylabel('R² Score')
                region_labels = ['Low M, Low R', 'Low M, High R', 'High M, Low R', 'High M, High R']
                ax.set_title(f'Performance: {region_labels[region]}')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
        
        # 13. Miles per day distribution
        ax13 = plt.subplot(4, 6, 13)
        ax13.hist(df['Miles_per_Day'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax13.axvline(x=100, color='red', linestyle='--', linewidth=2, label='100 miles/day')
        ax13.axvline(x=200, color='orange', linestyle='--', linewidth=2, label='200 miles/day')
        ax13.set_xlabel('Miles per Day')
        ax13.set_ylabel('Frequency')
        ax13.set_title('Miles per Day Distribution')
        ax13.legend()
        ax13.grid(True, alpha=0.3)
        
        # 14. Receipts per day distribution
        ax14 = plt.subplot(4, 6, 14)
        ax14.hist(df['Receipts_per_Day'], bins=50, alpha=0.7, color='green', edgecolor='black')
        ax14.axvline(x=100, color='red', linestyle='--', linewidth=2, label='$100/day')
        ax14.axvline(x=200, color='orange', linestyle='--', linewidth=2, label='$200/day')
        ax14.set_xlabel('Receipts per Day')
        ax14.set_ylabel('Frequency')
        ax14.set_title('Receipts per Day Distribution')
        ax14.legend()
        ax14.grid(True, alpha=0.3)
        
        # 15. Reimbursement per day distribution
        ax15 = plt.subplot(4, 6, 15)
        ax15.hist(df['Reimb_per_Day'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax15.axvline(x=200, color='red', linestyle='--', linewidth=2, label='$200/day')
        ax15.axvline(x=300, color='orange', linestyle='--', linewidth=2, label='$300/day')
        ax15.set_xlabel('Reimbursement per Day')
        ax15.set_ylabel('Frequency')
        ax15.set_title('Reimbursement per Day Distribution')
        ax15.legend()
        ax15.grid(True, alpha=0.3)
        
        # 16. Coefficient trends by day
        ax16 = plt.subplot(4, 6, 16)
        
        if 'daily_patterns' in self.analysis_results:
            days = []
            avg_a = []
            avg_b = []
            avg_c = []
            
            for day, data in self.analysis_results['daily_patterns']['by_day'].items():
                days.append(day)
                avg_a.append(data['avg_A_coef'])
                avg_b.append(data['avg_B_coef'])
                avg_c.append(data['avg_C_coef'])
            
            if days:
                ax16.plot(days, avg_a, 'o-', label='A (Days)', linewidth=2)
                ax16.plot(days, avg_b, 's-', label='B (Miles)', linewidth=2)
                ax16.plot(days, avg_c, '^-', label='C (Receipts)', linewidth=2)
                ax16.set_xlabel('Day')
                ax16.set_ylabel('Average Coefficient')
                ax16.set_title('Coefficient Trends by Day')
                ax16.legend()
                ax16.grid(True, alpha=0.3)
        
        # 17-20. Scatter plots for each region
        region_colors = ['blue', 'green', 'red', 'purple']
        region_labels = ['Low M, Low R', 'Low M, High R', 'High M, Low R', 'High M, High R']
        
        for i, region in enumerate(range(4)):
            ax = plt.subplot(4, 6, 17 + i)
            
            # Get data for this region across all days
            region_data = []
            for day in range(1, self.max_days + 1):
                df_region = df[(df['Days'] == day) & (df['Region'] == region)]
                if len(df_region) > 0:
                    region_data.append(df_region)
            
            if region_data:
                df_region_all = pd.concat(region_data)
                ax.scatter(df_region_all['Miles_per_Day'], df_region_all['Receipts_per_Day'], 
                          alpha=0.6, s=20, color=region_colors[i])
                ax.set_xlabel('Miles per Day')
                ax.set_ylabel('Receipts per Day')
                ax.set_title(f'{region_labels[i]} Scatter')
                ax.grid(True, alpha=0.3)
        
        # 21. Summary statistics
        ax21 = plt.subplot(4, 6, 21)
        ax21.axis('off')
        
        # Calculate summary statistics
        valid_results = [r for r in self.analysis_results['subregion_analysis'].values() 
                        if 'error' not in r and 'performance' in r]
        
        if valid_results:
            r2_scores = [r['performance']['r2_score'] for r in valid_results]
            
            summary_text = f"""
            Daily Rate Analysis Summary:
            
            Configuration:
            • Miles threshold: {self.miles_threshold}/day
            • Receipts threshold: ${self.receipts_threshold}/day
            • Days analyzed: 1-{self.max_days}
            
            Subregions:
            • Total possible: 56
            • Populated: {len(valid_results)}
            • Mean R²: {np.mean(r2_scores):.3f}
            • Best R²: {max(r2_scores):.3f}
            • Worst R²: {min(r2_scores):.3f}
            """
            
            ax21.text(0.1, 0.9, summary_text, transform=ax21.transAxes, 
                     fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # 22. Threshold analysis
        ax22 = plt.subplot(4, 6, 22)
        ax22.axis('off')
        
        if 'threshold_analysis' in self.analysis_results:
            caps = self.analysis_results['threshold_analysis']['reimbursement_caps']['potential_caps']
            total_records = len(df)
            
            threshold_text = f"""
            Reimbursement Threshold Analysis:
            
            Daily Reimbursement Caps:
            • ≤$200/day: {caps['daily_200']} ({caps['daily_200']/total_records*100:.1f}%)
            • ≤$300/day: {caps['daily_300']} ({caps['daily_300']/total_records*100:.1f}%)
            • ≤$400/day: {caps['daily_400']} ({caps['daily_400']/total_records*100:.1f}%)
            • ≤$500/day: {caps['daily_500']} ({caps['daily_500']/total_records*100:.1f}%)
            
            Max Daily Rates:
            • Miles: {df['Miles_per_Day'].max():.1f}/day
            • Receipts: ${df['Receipts_per_Day'].max():.1f}/day
            • Reimb: ${df['Reimb_per_Day'].max():.1f}/day
            """
            
            ax22.text(0.1, 0.9, threshold_text, transform=ax22.transAxes, 
                     fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # 23-24. Additional analysis plots
        ax23 = plt.subplot(4, 6, 23)
        
        # Performance vs sample size
        sample_sizes = []
        r2_scores = []
        
        for result in valid_results:
            sample_sizes.append(result['count'])
            r2_scores.append(result['performance']['r2_score'])
        
        if sample_sizes:
            ax23.scatter(sample_sizes, r2_scores, alpha=0.7, s=50)
            ax23.set_xlabel('Sample Size')
            ax23.set_ylabel('R² Score')
            ax23.set_title('Performance vs Sample Size')
            ax23.grid(True, alpha=0.3)
        
        ax24 = plt.subplot(4, 6, 24)
        
        # Region performance comparison
        if 'daily_patterns' in self.analysis_results:
            region_data = self.analysis_results['daily_patterns']['by_region']
            regions = list(region_data.keys())
            avg_r2 = [region_data[r]['avg_r2'] for r in regions]
            
            bars = ax24.bar(range(len(regions)), avg_r2, color=region_colors[:len(regions)], alpha=0.7)
            ax24.set_xlabel('Region Type')
            ax24.set_ylabel('Average R² Score')
            ax24.set_title('Performance by Region Type')
            ax24.set_xticks(range(len(regions)))
            ax24.set_xticklabels([region_labels[r] for r in regions], rotation=45)
            ax24.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, avg_r2):
                ax24.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                         f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'dashboard.png', 
                   dpi=100, bbox_inches='tight')
        plt.close()
        
        self.analysis_results['visualizations'].append('dashboard.png')
        
        return str(Path(output_dir) / 'dashboard.png')
    
    def print_comprehensive_results(self):
        """Print comprehensive daily rate analysis results."""
        print("\n" + "="*120)
        print("DAILY RATE ANALYSIS RESULTS (56 Subregions)")
        print("="*120)
        
        print(f"\nConfiguration:")
        print(f"Miles threshold: {self.miles_threshold} per day")
        print(f"Receipts threshold: ${self.receipts_threshold} per day")
        print(f"Days analyzed: 1-{self.max_days}")
        
        # Subregion results table
        print(f"\n{'Subregion':>10} {'Day':>4} {'Region':>8} {'Count':>6} {'A (Days)':>10} {'B (Miles)':>12} {'C (Receipts)':>14} {'R²':>8} {'RMSE':>10}")
        print("-" * 120)
        
        for subregion_id in range(56):
            result = self.analysis_results['subregion_analysis'].get(subregion_id, {})
            if 'error' not in result and 'coefficients' in result:
                day = result['day']
                region = result['region']
                count = result['count']
                A = result['coefficients']['A_days']
                B = result['coefficients']['B_miles']
                C = result['coefficients']['C_receipts']
                r2 = result['performance']['r2_score']
                rmse = result['performance']['rmse']
                
                print(f"{subregion_id:>10} {day:>4} {region:>8} {count:>6} {A:>10.3f} {B:>12.3f} {C:>14.3f} {r2:>8.3f} {rmse:>10.1f}")
            elif result.get('count', 0) == 0:
                day = (subregion_id // 4) + 1
                region = subregion_id % 4
                print(f"{subregion_id:>10} {day:>4} {region:>8} {'0':>6} {'N/A':>10} {'N/A':>12} {'N/A':>14} {'N/A':>8} {'N/A':>10}")
        
        # Performance ranking
        valid_results = [(subregion_id, result) for subregion_id, result in self.analysis_results['subregion_analysis'].items()
                        if 'error' not in result and 'performance' in result]
        
        if valid_results:
            sorted_results = sorted(valid_results, key=lambda x: x[1]['performance']['r2_score'], reverse=True)
            
            print(f"\nTop 10 Performing Subregions:")
            print(f"{'Rank':>4} {'Subregion':>10} {'Label':>25} {'R²':>8} {'Count':>6}")
            print("-" * 60)
            
            for i, (subregion_id, result) in enumerate(sorted_results[:10], 1):
                label = result['subregion_label']
                r2 = result['performance']['r2_score']
                count = result['count']
                print(f"{i:>4} {subregion_id:>10} {label:>25} {r2:>8.3f} {count:>6}")
        
        # Threshold analysis
        if 'threshold_analysis' in self.analysis_results:
            print(f"\nReimbursement Threshold Analysis:")
            caps = self.analysis_results['threshold_analysis']['reimbursement_caps']['potential_caps']
            total_records = sum(1 for r in self.analysis_results['subregion_analysis'].values() 
                              if 'count' in r for _ in range(r['count']))
            
            print(f"Daily reimbursement distribution:")
            print(f"≤$200/day: {caps['daily_200']} records")
            print(f"≤$300/day: {caps['daily_300']} records") 
            print(f"≤$400/day: {caps['daily_400']} records")
            print(f"≤$500/day: {caps['daily_500']} records")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive daily rate analysis report."""
        report = {
            'analysis_overview': {
                'title': 'Daily Rate Analysis (56 Subregions)',
                'configuration': self.analysis_results['configuration'],
                'timestamp': datetime.now().isoformat()
            },
            'subregion_analysis': self.analysis_results['subregion_analysis'],
            'threshold_analysis': self.analysis_results['threshold_analysis'],
            'daily_patterns': self.analysis_results['daily_patterns'],
            'visualizations': self.analysis_results['visualizations']
        }
        
        return report


def main():
    parser = argparse.ArgumentParser(description='Daily Rate Analysis (56 Subregions)')
    parser.add_argument('--csv', default='public.csv', help='Input CSV file (default: public.csv)')
    parser.add_argument('--output-dir', default=None, 
                       help='Output directory (default: auto-generated series directory)')
    parser.add_argument('--miles-threshold', type=int, default=100,
                       help='Miles per day threshold (default: 100)')
    parser.add_argument('--receipts-threshold', type=int, default=100,
                       help='Receipts per day threshold (default: 100)')
    parser.add_argument('--series-mode', action='store_true',
                       help='Use series-based output organization (recommended)')
    parser.add_argument('--series-id', default=None,
                       help='Series identifier (default: current date)')
    
    args = parser.parse_args()
    
    print("Daily Rate Analysis (56 Subregions)")
    print("=" * 50)
    
    # Import series utilities
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from core.utils import create_series_output_dir, create_timestamped_output_dir
    
    # Create output directory
    if args.output_dir is None:
        if args.series_mode:
            # Use new series-based organization
            series_name = f"daily_rate_m{args.miles_threshold}_r{args.receipts_threshold}"
            args.output_dir = create_series_output_dir(series_name, args.series_id)
            print(f"Using series mode: {args.output_dir}")
        else:
            # Use legacy timestamped format
            args.output_dir = create_timestamped_output_dir("daily_rate_analysis")
            print(f"Using legacy mode: {args.output_dir}")
    else:
        # User-specified directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = DailyRateAnalyzer(args.miles_threshold, args.receipts_threshold)
    
    # Load and prepare data
    df = analyzer.load_and_prepare_data(args.csv)
    
    # Perform comprehensive analysis
    analyzer.analyze_all_subregions(df)
    analyzer.analyze_reimbursement_thresholds(df)
    analyzer.analyze_daily_patterns(df)
    
    # Create visualizations
    viz_file = analyzer.create_comprehensive_visualizations(df, args.output_dir)
    print(f"Visualization saved to: {viz_file}")
    
    # Print results
    analyzer.print_comprehensive_results()
    
    # Generate and save report
    report = analyzer.generate_comprehensive_report()
    
    output_file = Path(args.output_dir) / 'results.json'
    with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDaily rate analysis results saved to: {output_file}")
    
    return analyzer, df, report


if __name__ == "__main__":
    main() 