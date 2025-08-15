#!/usr/bin/env python3
"""
Unified Comparison Script for Orientation Detection
==================================================

Compare ML approaches and model architectures in a unified interface.
Supports matrix comparisons with beautiful overview output.

Usage:
    python compare.py                                    # Compare all approaches on all models
    python compare.py --approaches unit_vector cgd       # Compare specific approaches on all models
    python compare.py --models tf_efficientnetv2_b0.in1k # Compare all approaches on specific model
    python compare.py --approaches cgd --models tf_efficientnetv2_b0.in1k --epochs 10 --batch-size 32
"""

import os
import subprocess
import json
import time
import argparse
import shutil
from datetime import datetime
from typing import Dict, List, Tuple
from architectures import get_enabled_architectures, get_default_learning_rate, get_scaled_learning_rate

# Default loss types for each approach and approach-loss combinations
DEFAULT_APPROACH_LOSSES = {
    # Base approaches with default loss types
    "direct_angle": "mse", 
    "unit_vector": "mse",
    "psc": "mse",
    "cgd": "kl_divergence",    
    # Classification approach variants
    "classification": "cross_entropy",           # Default classification
    "classification_csl": "csl",                # Circular Smooth Label variant
    "classification_dcl": "dcl",                # Dense Coded Labels variant
    "multibin": "multibin",
}

def parse_base_approach(approach: str) -> str:
    """Extract base approach from approach-loss combination.
    
    Examples:
        classification_csl -> classification
        classification_dcl -> classification  
        cgd -> cgd
    """
    # Known base approaches
    base_approaches = ["classification", "unit_vector", "direct_angle", "psc", "cgd", "multibin"]
    
    # Check if it's already a base approach
    if approach in base_approaches:
        return approach
        
    # Check for approach_variant pattern
    for base in base_approaches:
        if approach.startswith(f"{base}_"):
            return base
    
    # Fallback - return as-is
    return approach

def run_experiment(approach: str, model_name: str, epochs: int, batch_size: int, timestamp: str, mixed_precision: bool = False) -> Dict:
    """Run single training+testing experiment using train.py."""
    loss_type = DEFAULT_APPROACH_LOSSES[approach]
    learning_rate = get_scaled_learning_rate(model_name, batch_size)
    
    # Parse approach-loss combinations (e.g., "classification_csl" -> "classification")
    base_approach = parse_base_approach(approach)
    
    print(f"  Running {approach} + {model_name.split('.')[0]} (LR: {learning_rate:.0e})")
    
    # Create unique experiment directory
    model_short = model_name.split('.')[0]
    exp_dir = f"{timestamp}_compare_{approach}_{model_short}"
    weights_dir = f"comparison/{exp_dir}"
    
    try:
        # Use train.py with --run-test for automatic training + testing
        cmd = [
            "python", "train.py",
            "--approach", base_approach,
            "--loss-type", loss_type,
            "--max-epochs", str(epochs),
            "--learning-rate", str(learning_rate),
            "--batch-size", str(batch_size),
            "--train-dir", "data/datasets/train_drcd",
            "--save-dir", weights_dir,
            "--model-name", model_name,
            "--run-test",
            "--test-dirs", "data/datasets/test_drdc",
        ]
        
        # Add mixed precision flag if enabled
        if mixed_precision:
            cmd.append("--mixed-precision")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True) # timeout=90*60
        training_time = time.time() - start_time
        
        if result.returncode == 0:
            # Extract test results from structured output
            test_results = extract_test_results(result.stdout)
            
            # Find test MAE for summary
            test_mae = None
            for key, value in test_results.items():
                if 'mae_deg' in key.lower() and isinstance(value, (int, float)):
                    test_mae = value
                    break
            
            return {
                'approach': approach,
                'model_name': model_name,
                'model_short': model_short,
                'loss_type': loss_type,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'training_time': training_time,
                'test_mae': test_mae,
                'success': True,
                **test_results
            }
        else:
            return {
                'approach': approach,
                'model_name': model_name,
                'model_short': model_short,
                'success': False,
                'training_time': training_time,
                'error': result.stderr[-200:] if result.stderr else 'Unknown error'
            }
            
    except subprocess.TimeoutExpired:
        return {
            'approach': approach,
            'model_name': model_name,
            'model_short': model_short,
            'success': False,
            'error': 'timeout'
        }
    except Exception as e:
        return {
            'approach': approach,
            'model_name': model_name,
            'model_short': model_short,
            'success': False,
            'error': str(e)
        }
    finally:
        # Cleanup weights directory - COMMENTED OUT to preserve comparison models for reuse
        # if os.path.exists(weights_dir):
        #     shutil.rmtree(weights_dir, ignore_errors=True)
        pass

def extract_test_results(stdout: str) -> Dict:
    """Extract test results from train.py structured output."""
    results = {}
    lines = stdout.split('\n')
    capturing = False
    
    for line in lines:
        if "=== TEST_RESULTS_START ===" in line:
            capturing = True
            continue
        elif "=== TEST_RESULTS_END ===" in line:
            capturing = False
            continue
        elif capturing and "=" in line:
            key, value = line.split("=", 1)
            try:
                results[key] = float(value)
            except ValueError:
                results[key] = value
    
    return results

def print_results_matrix(results: List[Dict], approaches: List[str], models: List[str], output_file=None):
    """Print beautiful results matrix with MAE scores."""
    content = []
    content.append(f"\nRESULTS MATRIX (Test MAE in degrees)")
    content.append("=" * 100)
    
    # Create results lookup
    results_lookup = {}
    for r in results:
        if r['success'] and r.get('test_mae') is not None:
            results_lookup[(r['approach'], r['model_name'])] = r['test_mae']
    
    # Print header
    model_shorts = [m.split('.')[0][:15] for m in models]  # Truncate long names
    header = f"{'Approach':<12}"
    for model_short in model_shorts:
        header += f" | {model_short:>8}"
    content.append(header)
    content.append("-" * len(header))
    
    # Print matrix rows
    for approach in approaches:
        row = f"{approach:<12}"
        for model in models:
            mae = results_lookup.get((approach, model))
            if mae is not None:
                row += f" | {mae:>7.1f}°"
            else:
                row += f" | {'FAIL':>8}"
        content.append(row)
    
    # Print to console
    for line in content:
        print(line)
    
    # Write to file if specified
    if output_file:
        output_file.write('\n'.join(content) + '\n')

def print_best_performers(results: List[Dict], output_file=None):
    """Print best performers analysis."""
    successful = [r for r in results if r['success'] and r.get('test_mae') is not None]
    
    if not successful:
        content = ["\nNo successful experiments to analyze."]
        for line in content:
            print(line)
        if output_file:
            output_file.write('\n'.join(content) + '\n')
        return
    
    content = []
    content.append(f"\nBEST PERFORMERS")
    content.append("=" * 50)
    
    # Overall best
    best_overall = min(successful, key=lambda x: x['test_mae'])
    content.append(f"Overall Best: {best_overall['approach'].upper()} + {best_overall['model_short']}")
    content.append(f"  Test MAE: {best_overall['test_mae']:.1f}°, Training Time: {best_overall['training_time']:.0f}s")
    
    # Fastest
    fastest = min(successful, key=lambda x: x['training_time'])
    content.append(f"Fastest: {fastest['approach'].upper()} + {fastest['model_short']}")
    content.append(f"  Test MAE: {fastest['test_mae']:.1f}°, Training Time: {fastest['training_time']:.0f}s")
    
    # Best per approach
    content.append(f"\nBest Model per Approach:")
    approaches = set(r['approach'] for r in successful)
    for approach in sorted(approaches):
        approach_results = [r for r in successful if r['approach'] == approach]
        if approach_results:
            best = min(approach_results, key=lambda x: x['test_mae'])
            content.append(f"  {approach:<12}: {best['model_short']} ({best['test_mae']:.1f}°)")
    
    # Best per model
    content.append(f"\nBest Approach per Model:")
    models = set(r['model_name'] for r in successful)
    for model in sorted(models):
        model_results = [r for r in successful if r['model_name'] == model]
        if model_results:
            best = min(model_results, key=lambda x: x['test_mae'])
            model_short = model.split('.')[0]
            content.append(f"  {model_short:<20}: {best['approach']} ({best['test_mae']:.1f}°)")
    
    # Print to console
    for line in content:
        print(line)
    
    # Write to file if specified
    if output_file:
        output_file.write('\n'.join(content) + '\n')

def print_speed_ranking(results: List[Dict], output_file=None):
    """Print speed ranking of successful experiments."""
    successful = [r for r in results if r['success']]
    
    if not successful:
        return
    
    content = []
    content.append(f"\nSPEED RANKING (Top 10 Fastest)")
    content.append("=" * 60)
    content.append(f"{'Rank':<4} {'Approach':<12} {'Model':<20} {'Time':<6} {'MAE':<8}")
    content.append("-" * 60)
    
    # Sort by training time
    by_speed = sorted(successful, key=lambda x: x['training_time'])
    
    for i, result in enumerate(by_speed[:10], 1):
        mae_str = f"{result.get('test_mae', 0):.1f}°" if result.get('test_mae') else "N/A"
        content.append(f"{i:<4} {result['approach']:<12} {result['model_short']:<20} {result['training_time']:<5.0f}s {mae_str:<8}")
    
    # Print to console
    for line in content:
        print(line)
    
    # Write to file if specified
    if output_file:
        output_file.write('\n'.join(content) + '\n')

def print_recommendations(results: List[Dict], output_file=None):
    """Print actionable recommendations."""
    successful = [r for r in results if r['success'] and r.get('test_mae') is not None]
    
    if not successful:
        return
    
    content = []
    content.append(f"\nRECOMMENDATIONS")
    content.append("=" * 50)
    
    # Best overall
    best = min(successful, key=lambda x: x['test_mae'])
    content.append(f"Best Accuracy:")
    content.append(f"  python train.py --approach={best['approach']} --model-name={best['model_name']}")
    
    # Best speed/accuracy tradeoff (within 10% of best accuracy but much faster)
    best_mae = best['test_mae']
    threshold = best_mae * 1.1  # Within 10% of best accuracy
    
    candidates = [r for r in successful if r['test_mae'] <= threshold]
    if len(candidates) > 1:
        fastest_good = min(candidates, key=lambda x: x['training_time'])
        if fastest_good != best:
            speedup = best['training_time'] / fastest_good['training_time']
            content.append(f"Best Speed/Accuracy Tradeoff ({speedup:.1f}x faster, {fastest_good['test_mae']:.1f}° vs {best['test_mae']:.1f}°):")
            content.append(f"  python train.py --approach={fastest_good['approach']} --model-name={fastest_good['model_name']}")
    
    # Print to console
    for line in content:
        print(line)
    
    # Write to file if specified
    if output_file:
        output_file.write('\n'.join(content) + '\n')

def print_summary_statistics(results: List[Dict], output_file=None):
    """Print summary statistics."""
    total = len(results)
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    content = []
    content.append(f"\nSUMMARY STATISTICS")
    content.append("=" * 40)
    content.append(f"Total Experiments: {total}")
    content.append(f"Successful: {len(successful)} ({len(successful)/total*100:.1f}%)")
    content.append(f"Failed: {len(failed)} ({len(failed)/total*100:.1f}%)")
    
    if successful:
        training_times = [r['training_time'] for r in successful]
        content.append(f"Average Training Time: {sum(training_times)/len(training_times):.0f}s")
        content.append(f"Total Training Time: {sum(training_times)/3600:.1f} hours")
        
        test_maes = [r['test_mae'] for r in successful if r.get('test_mae')]
        if test_maes:
            content.append(f"Best Test MAE: {min(test_maes):.1f}°")
            content.append(f"Average Test MAE: {sum(test_maes)/len(test_maes):.1f}°")
    
    if failed:
        content.append(f"\nFailed Experiments:")
        for result in failed[:5]:  # Show first 5 failures
            content.append(f"  {result['approach']} + {result['model_short']}: {result.get('error', 'unknown')}")
        if len(failed) > 5:
            content.append(f"  ... and {len(failed)-5} more")
    
    # Print to console
    for line in content:
        print(line)
    
    # Write to file if specified
    if output_file:
        output_file.write('\n'.join(content) + '\n')

def main():
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    parser = argparse.ArgumentParser(description="Unified comparison of approaches and models")
    parser.add_argument("--approaches", nargs='+', 
                       choices=list(DEFAULT_APPROACH_LOSSES.keys()),
                       help="Approaches to test. Supports approach-loss combinations like 'classification_csl'. (default: all)")
    parser.add_argument("--models", nargs='+', 
                       help="Models to test (default: all from architectures.py)")
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Epochs per experiment")
    parser.add_argument("--batch-size", type=int, default=16, 
                       help="Batch size for training")
    parser.add_argument("--output", type=str, default=f"{timestamp}_comparison_results.json", 
                       help="Output JSON file")
    parser.add_argument("--mixed-precision", action="store_true",
                       help="Enable mixed precision training (16-bit) for faster training and lower memory usage")
    
    args = parser.parse_args()
    
    # Determine approaches and models to test
    approaches = args.approaches or list(DEFAULT_APPROACH_LOSSES.keys())
    models = args.models or get_enabled_architectures()
    
    total_experiments = len(approaches) * len(models)
    
    print("UNIFIED COMPARISON EXPERIMENT")
    print("=" * 50)
    print(f"Approaches: {', '.join(approaches)}")
    print(f"Models: {len(models)} models")
    print(f"Total Experiments: {total_experiments}")
    print(f"Epochs per Experiment: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Mixed Precision: {'Enabled' if args.mixed_precision else 'Disabled'}")
    print()
    
    # Run all experiments
    results = []
    experiment_count = 0
    
    for approach in approaches:
        for model in models:
            experiment_count += 1
            print(f"Experiment {experiment_count}/{total_experiments}")
            
            result = run_experiment(approach, model, args.epochs, args.batch_size, timestamp, args.mixed_precision)
            results.append(result)
            
            # Brief status
            if result['success']:
                mae_str = f", MAE: {result.get('test_mae', 0):.1f}°" if result.get('test_mae') else ""
                print(f"  ✓ Success ({result['training_time']:.0f}s{mae_str})")
            else:
                print(f"  ✗ Failed: {result.get('error', 'unknown')}")
            
            time.sleep(1)  # Brief pause
    
    # Save detailed results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print beautiful overview and save to file
    print("\n" + "="*100)
    print("COMPARISON RESULTS OVERVIEW")
    print("="*100)
    
    # Create overview file
    overview_filename = f"{timestamp}_comparison_overview.txt"
    with open(overview_filename, 'w') as overview_file:
        # Write header to file
        overview_file.write("="*100 + "\n")
        overview_file.write("COMPARISON RESULTS OVERVIEW\n")
        overview_file.write("="*100 + "\n")
        
        # Generate and write all sections
        print_results_matrix(results, approaches, models, overview_file)
        print_best_performers(results, overview_file)
        print_speed_ranking(results, overview_file)
        print_recommendations(results, overview_file)
        print_summary_statistics(results, overview_file)
    
    print(f"\nDetailed results saved to: {args.output}")
    print(f"Overview saved to: {overview_filename}")

if __name__ == "__main__":
    main()