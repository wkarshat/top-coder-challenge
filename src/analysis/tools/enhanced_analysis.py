#!/usr/bin/env python3
"""
Enhanced Pattern Discovery Analysis

Focuses on uncovering patterns beyond the basic linear formula through:
- Interaction terms (Miles*Days, Receipts*Days, etc.)
- Ratio features (Miles/Days, Receipts/Days)
- Residual analysis
- Threshold detection
- Advanced correlation analysis
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_data(csv_file):
    """Load and prepare data with interaction and ratio features."""
    df = pd.read_csv(csv_file)
    
    # Convert to numeric
    for col in ['Days', 'Miles', 'Receipts', 'Reimb']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    # Create interaction features
    df['Miles_x_Days'] = df['Miles'] * df['Days']
    df['Receipts_x_Days'] = df['Receipts'] * df['Days']
    df['Miles_x_Receipts'] = df['Miles'] * df['Receipts']
    
    # Create ratio features (avoid division by zero)
    df['Miles_per_Day'] = df['Miles'] / df['Days']
    df['Receipts_per_Day'] = df['Receipts'] / df['Days']
    df['Miles_per_Receipt'] = df['Miles'] / (df['Receipts'] + 1e-6)  # Small epsilon
    
    return df


def test_formula_accuracy(df):
    """Test how well the simple formula fits the data."""
    # Calculate predicted values using the formula
    df['Formula_Pred'] = df['Days'] * 100 + df['Miles'] * 0.5 + df['Receipts']
    
    # Calculate R² and residuals
    r2 = r2_score(df['Reimb'], df['Formula_Pred'])
    residuals = df['Reimb'] - df['Formula_Pred']
    rmse = np.sqrt(mean_squared_error(df['Reimb'], df['Formula_Pred']))
    
    return r2, residuals, rmse

def analyze_interactions(df):
    """Analyze interaction and ratio feature correlations."""
    interaction_features = ['Miles_x_Days', 'Receipts_x_Days', 'Miles_x_Receipts',
                           'Miles_per_Day', 'Receipts_per_Day', 'Miles_per_Receipt']
    
    correlations = {}
    for feature in interaction_features:
        corr, p_val = pearsonr(df[feature], df['Reimb'])
        correlations[feature] = {'correlation': corr, 'p_value': p_val}
    
    return correlations

def test_enhanced_models(df):
    """Test models with interaction terms and polynomial features."""
    base_features = ['Days', 'Miles', 'Receipts']
    interaction_features = ['Miles_x_Days', 'Receipts_x_Days', 'Miles_x_Receipts']
    ratio_features = ['Miles_per_Day', 'Receipts_per_Day']
    
    results = {}
    
    # 1. Base linear model (formula)
    X_base = df[base_features]
    y = df['Reimb']
    
    model_base = LinearRegression()
    model_base.fit(X_base, y)
    pred_base = model_base.predict(X_base)
    results['Linear_Base'] = r2_score(y, pred_base)
    
    # 2. Linear with interactions
    X_interact = df[base_features + interaction_features]
    model_interact = LinearRegression()
    model_interact.fit(X_interact, y)
    pred_interact = model_interact.predict(X_interact)
    results['Linear_Interactions'] = r2_score(y, pred_interact)
    
    # 3. Linear with ratios
    X_ratio = df[base_features + ratio_features]
    model_ratio = LinearRegression()
    model_ratio.fit(X_ratio, y)
    pred_ratio = model_ratio.predict(X_ratio)
    results['Linear_Ratios'] = r2_score(y, pred_ratio)
    
    # 4. Linear with all features
    X_all = df[base_features + interaction_features + ratio_features]
    model_all = LinearRegression()
    model_all.fit(X_all, y)
    pred_all = model_all.predict(X_all)
    results['Linear_All'] = r2_score(y, pred_all)
    
    # 5. Polynomial features (degree 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_base)
    model_poly = LinearRegression()
    model_poly.fit(X_poly, y)
    pred_poly = model_poly.predict(X_poly)
    results['Polynomial_2'] = r2_score(y, pred_poly)
    
    # 6. Random Forest (non-linear)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_base, y)
    pred_rf = rf.predict(X_base)
    results['Random_Forest'] = r2_score(y, pred_rf)
    
    return results

def analyze_residuals(df, residuals):
    """Analyze residual patterns to identify systematic deviations."""
    df_res = df.copy()
    df_res['Residuals'] = residuals
    
    # Residual patterns by Days
    day_residuals = df_res.groupby('Days')['Residuals'].agg(['mean', 'std', 'count'])
    
    # Residual correlations with features
    residual_corrs = {}
    for col in ['Days', 'Miles', 'Receipts', 'Miles_per_Day', 'Receipts_per_Day']:
        corr, p_val = pearsonr(df_res[col], df_res['Residuals'])
        residual_corrs[col] = {'correlation': corr, 'p_value': p_val}
    
    return day_residuals, residual_corrs

def detect_thresholds(df):
    """Detect potential threshold effects in the data."""
    thresholds = {}
    
    # Days thresholds (mentioned 5-day bonus in interviews)
    for threshold in [3, 5, 7, 10]:
        short_trips = df[df['Days'] <= threshold]
        long_trips = df[df['Days'] > threshold]
        
        if len(short_trips) > 10 and len(long_trips) > 10:
            # Test if different patterns exist
            short_r2, _, _ = test_formula_accuracy(short_trips)
            long_r2, _, _ = test_formula_accuracy(long_trips)
            
            thresholds[f'Days_{threshold}'] = {
                'short_trips_r2': short_r2,
                'long_trips_r2': long_r2,
                'short_count': len(short_trips),
                'long_count': len(long_trips)
            }
    
    return thresholds

def create_enhanced_visualizations(df, residuals, output_dir):
    """Create visualizations for pattern discovery."""
    plt.style.use('default')
    
    # 1. Residual analysis plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs predicted
    predicted = df['Days'] * 100 + df['Miles'] * 0.5 + df['Receipts']
    ax1.scatter(predicted, residuals, alpha=0.6, s=20)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Reimb')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')
    
    # Residuals by Days
    days_unique = sorted(df['Days'].unique())
    residual_by_day = [residuals[df['Days'] == day] for day in days_unique]
    ax2.boxplot(residual_by_day, labels=[f'D{d}' for d in days_unique])
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals by Trip Duration')
    ax2.tick_params(axis='x', rotation=45)
    
    # Interaction feature correlations
    interactions = ['Miles_x_Days', 'Receipts_x_Days', 'Miles_per_Day', 'Receipts_per_Day']
    corrs = [pearsonr(df[feat], df['Reimb'])[0] for feat in interactions]
    ax3.bar(range(len(interactions)), corrs)
    ax3.set_xticks(range(len(interactions)))
    ax3.set_xticklabels([f.replace('_', '\n') for f in interactions], rotation=45)
    ax3.set_ylabel('Correlation with Reimb')
    ax3.set_title('Interaction Feature Correlations')
    
    # Miles/Day vs Receipts/Day colored by residuals
    scatter = ax4.scatter(df['Miles_per_Day'], df['Receipts_per_Day'], 
                         c=residuals, cmap='RdBu_r', alpha=0.6, s=20)
    ax4.set_xlabel('Miles per Day')
    ax4.set_ylabel('Receipts per Day')
    ax4.set_title('Ratios colored by Residuals')
    plt.colorbar(scatter, ax=ax4, label='Residuals')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'pattern_discovery.png', dpi=100, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Enhanced Pattern Discovery Analysis')
    parser.add_argument('--csv', default='public.csv', help='Input CSV file (default: public.csv)')
    parser.add_argument('--output-dir', default='outputs/analysis_143208', 
                       help='Output directory')
    args = parser.parse_args()
    
    print("Enhanced Pattern Discovery Analysis")
    print("=" * 50)
    
    # Load data with enhanced features
    df = load_data(args.csv)
    print(f"Loaded {len(df)} records with enhanced features")
    
    # Test formula accuracy
    formula_r2, residuals, rmse = test_formula_accuracy(df)
    print(f"\nFormula Performance:")
    print(f"  R² = {formula_r2:.4f}")
    print(f"  RMSE = {rmse:.2f}")
    
    # Analyze interactions
    interaction_corrs = analyze_interactions(df)
    print(f"\nInteraction Feature Correlations:")
    for feature, stats in interaction_corrs.items():
        print(f"  {feature}: r = {stats['correlation']:.3f} (p = {stats['p_value']:.3e})")
    
    # Test enhanced models
    model_results = test_enhanced_models(df)
    print(f"\nModel Performance Comparison:")
    for model_name, r2 in model_results.items():
        improvement = ((r2 - formula_r2) / formula_r2) * 100
        print(f"  {model_name}: R² = {r2:.4f} ({improvement:+.1f}%)")
    
    # Analyze residuals
    day_residuals, residual_corrs = analyze_residuals(df, residuals)
    print(f"\nResidual Analysis:")
    print(f"  Mean absolute residual: {np.abs(residuals).mean():.2f}")
    print(f"  Residual std: {residuals.std():.2f}")
    
    # Detect thresholds
    thresholds = detect_thresholds(df)
    print(f"\nThreshold Analysis:")
    for threshold, stats in thresholds.items():
        print(f"  {threshold}: Short R² = {stats['short_trips_r2']:.3f}, "
              f"Long R² = {stats['long_trips_r2']:.3f}")
    
    # Create visualizations
    create_enhanced_visualizations(df, residuals, args.output_dir)
    print(f"\nPattern discovery plot saved to: {args.output_dir}/pattern_discovery.png")
    
    # Summary insights
    print(f"\nKey Insights:")
    best_interaction = max(interaction_corrs.items(), key=lambda x: abs(x[1]['correlation']))
    print(f"  Strongest interaction: {best_interaction[0]} (r = {best_interaction[1]['correlation']:.3f})")
    
    best_model_name = max(model_results.items(), key=lambda x: x[1])
    improvement = ((best_model_name[1] - formula_r2) / formula_r2) * 100
    print(f"  Best model: {best_model_name[0]} (+{improvement:.1f}% over formula)")
    
    if np.abs(residuals).mean() > 50:
        print(f"  Large residuals suggest systematic patterns beyond linear formula")
    
    return df, residuals, model_results

if __name__ == "__main__":
    main()
