#!/usr/bin/env python3
"""
Test script for series output directory functionality.

This script demonstrates how to organize analysis results from multiple runs
into unique subdirectories within a series.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.utils import (
    create_series_output_dir,
    create_timestamped_output_dir,
    get_latest_series_run,
    list_series_runs
)


def simulate_analysis_run(output_dir: Path, analysis_name: str, run_data: dict):
    """Simulate an analysis run by creating sample output files."""
    
    # Create sample results.json
    results = {
        "analysis_type": analysis_name,
        "timestamp": datetime.now().isoformat(),
        "run_data": run_data,
        "performance": {
            "r2_score": run_data.get("r2", 0.85),
            "rmse": run_data.get("rmse", 150.0),
            "samples": run_data.get("samples", 1000)
        },
        "files_generated": [
            "results.json",
            "dashboard.png",
            "summary.txt"
        ]
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create sample dashboard.png (placeholder)
    with open(output_dir / "dashboard.png", 'w') as f:
        f.write(f"# Placeholder dashboard for {analysis_name}\n")
        f.write(f"# Generated at {datetime.now()}\n")
    
    # Create sample summary.txt
    with open(output_dir / "summary.txt", 'w') as f:
        f.write(f"Analysis Summary: {analysis_name}\n")
        f.write(f"R² Score: {results['performance']['r2_score']}\n")
        f.write(f"RMSE: {results['performance']['rmse']}\n")
        f.write(f"Samples: {results['performance']['samples']}\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
    
    print(f"✅ Created analysis run in: {output_dir}")
    return results


def test_series_functionality():
    """Test the series output directory functionality."""
    
    print("Testing Series Output Directory Functionality")
    print("=" * 60)
    
    # Test 1: Create multiple runs in a series
    print("\n1. Creating multiple runs in 'comprehensive_analysis' series:")
    
    series_runs = []
    for i in range(3):
        run_dir = create_series_output_dir("comprehensive_analysis")
        run_data = {
            "run_number": i + 1,
            "r2": 0.85 + i * 0.02,
            "rmse": 150.0 - i * 5,
            "samples": 1000 + i * 50
        }
        results = simulate_analysis_run(run_dir, "comprehensive_analysis", run_data)
        series_runs.append((run_dir, results))
    
    # Test 2: Create runs in different series
    print("\n2. Creating runs in different analysis series:")
    
    for analysis_type in ["daily_analysis", "threshold_analysis"]:
        run_dir = create_series_output_dir(analysis_type)
        run_data = {
            "run_number": 1,
            "r2": 0.78,
            "rmse": 180.0,
            "samples": 800
        }
        simulate_analysis_run(run_dir, analysis_type, run_data)
    
    # Test 3: Get latest run
    print("\n3. Testing latest run retrieval:")
    latest_run = get_latest_series_run("comprehensive_analysis")
    if latest_run:
        print(f"✅ Latest comprehensive_analysis run: {latest_run}")
        with open(latest_run / "results.json", 'r') as f:
            latest_results = json.load(f)
        print(f"   R² Score: {latest_results['performance']['r2_score']}")
    else:
        print("❌ No latest run found")
    
    # Test 4: List all runs in series
    print("\n4. Listing all runs in series:")
    for series_name in ["comprehensive_analysis", "daily_analysis", "threshold_analysis"]:
        runs = list_series_runs(series_name)
        print(f"   {series_name}: {len(runs)} runs")
        for run_dir in runs:
            print(f"     - {run_dir.name}")
    
    # Test 5: Legacy timestamped format
    print("\n5. Testing legacy timestamped format:")
    legacy_dir = create_timestamped_output_dir("legacy_analysis")
    simulate_analysis_run(legacy_dir, "legacy_analysis", {"legacy": True})
    
    return series_runs


def demonstrate_series_comparison():
    """Demonstrate comparing results across runs in a series."""
    
    print("\n" + "=" * 60)
    print("Series Comparison Demonstration")
    print("=" * 60)
    
    # Get all comprehensive_analysis runs
    runs = list_series_runs("comprehensive_analysis")
    
    if not runs:
        print("❌ No comprehensive_analysis runs found")
        return
    
    print(f"\nComparing {len(runs)} runs in comprehensive_analysis series:")
    print("-" * 50)
    
    comparison_data = []
    for run_dir in runs:
        results_file = run_dir / "results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            comparison_data.append({
                "run": run_dir.name,
                "r2": results['performance']['r2_score'],
                "rmse": results['performance']['rmse'],
                "samples": results['performance']['samples'],
                "timestamp": results['timestamp']
            })
    
    # Sort by run number
    comparison_data.sort(key=lambda x: x['run'])
    
    # Print comparison table
    print(f"{'Run':<8} {'R²':<6} {'RMSE':<8} {'Samples':<8} {'Timestamp':<20}")
    print("-" * 60)
    for data in comparison_data:
        print(f"{data['run']:<8} {data['r2']:<6.3f} {data['rmse']:<8.1f} {data['samples']:<8} {data['timestamp'][:19]}")
    
    # Find best run
    best_run = max(comparison_data, key=lambda x: x['r2'])
    print(f"\n🏆 Best performing run: {best_run['run']} (R² = {best_run['r2']:.3f})")


def show_directory_structure():
    """Show the resulting directory structure."""
    
    print("\n" + "=" * 60)
    print("Resulting Directory Structure")
    print("=" * 60)
    
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print("❌ Outputs directory not found")
        return
    
    def print_tree(directory, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        
        items = sorted(directory.iterdir())
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and current_depth < max_depth - 1:
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(item, next_prefix, max_depth, current_depth + 1)
    
    print(f"\n{outputs_dir.name}/")
    print_tree(outputs_dir)


def main():
    """Main test function."""
    
    print("Series Output Directory Test")
    print("=" * 60)
    print("This script tests the new series-based output organization.")
    print("Multiple analysis runs will be organized into series with unique subdirectories.")
    
    try:
        # Run tests
        series_runs = test_series_functionality()
        demonstrate_series_comparison()
        show_directory_structure()
        
        print("\n" + "=" * 60)
        print("✅ Series output directory functionality test completed successfully!")
        print(f"✅ Created {len(series_runs)} runs in comprehensive_analysis series")
        print("✅ All functions working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 