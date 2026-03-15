#!/usr/bin/env python3
"""
Unified Comparison Script for Orientation Estimation
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
import math
from datetime import datetime
from typing import Dict, List, Optional
from architectures import get_enabled_architectures, get_default_learning_rate, get_scaled_learning_rate

# Base approaches supported by the training system
BASE_APPROACHES = ["direct_angle", "unit_vector", "psc", "cgd", "classification", ] # "multibin"

def parse_approach_and_loss(approach_str: str) -> tuple[str, str | None]:
    """Parse approach string into base approach and loss type.
    
    Examples:
        direct_angle_mae -> ("direct_angle", "mae")
        classification_csl -> ("classification", "csl")  
        cgd -> ("cgd", None)
    """
    if "_" in approach_str:
        base, loss = approach_str.rsplit("_", 1)
        if base in BASE_APPROACHES:
            return base, loss
    
    # If no underscore or base not recognized, treat as base approach
    if approach_str in BASE_APPROACHES:
        return approach_str, None
    
    raise ValueError(f"Unknown approach '{approach_str}'. Valid base approaches: {BASE_APPROACHES}")

def _run_single(approach: str, model_name: str, epochs: int, batch_size: int,
                timestamp: str, run_idx: int, mixed_precision: bool = False,
                keep_checkpoints: str = 'none') -> Dict:
    """Run one training+testing experiment for a given random seed (run_idx)."""
    base_approach, loss_type = parse_approach_and_loss(approach)
    learning_rate = get_scaled_learning_rate(model_name, batch_size)
    model_short = model_name.split('.')[0]
    exp_dir = f"{timestamp}_compare_{approach}_{model_short}_run{run_idx}"
    weights_dir = f"comparison/{exp_dir}"

    try:
        cmd = [
            "python", "train.py",
            "--approach", base_approach,
            "--max-epochs", str(epochs),
            "--learning-rate", str(learning_rate),
            "--batch-size", str(batch_size),
            "--train-dir", "data/datasets/train_drcd",
            "--save-dir", weights_dir,
            "--model-name", model_name,
            "--random-seed", str(run_idx),  # Different seed per run → different train/val split
            "--run-test",
            "--test-dirs", "data/datasets/test_drcd",
        ]

        if loss_type is not None:
            cmd.extend(["--loss-type", loss_type])
        if mixed_precision:
            cmd.append("--mixed-precision")

        # Always use compact checkpoints in comparison mode
        cmd.extend(["--save-weights-only", "--no-save-last"])

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        training_time = time.time() - start_time

        if result.returncode == 0:
            test_results = extract_test_results(result.stdout)
            test_mae = next(
                (v for k, v in test_results.items()
                 if 'mae_deg' in k.lower() and isinstance(v, (int, float))),
                None
            )
            if test_mae is None or (isinstance(test_mae, float) and math.isnan(test_mae)):
                return {
                    'approach': approach,
                    'model_name': model_name,
                    'model_short': model_short,
                    'run_idx': run_idx,
                    'success': False,
                    'training_time': training_time,
                    'weights_dir': weights_dir,
                    'error': 'No valid test MAE — model may have diverged (NaN) or test dir missing'
                }
            return {
                'approach': approach,
                'model_name': model_name,
                'model_short': model_short,
                'loss_type': loss_type,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'run_idx': run_idx,
                'training_time': training_time,
                'test_mae': test_mae,
                'weights_dir': weights_dir,
                'success': True,
                **test_results
            }
        else:
            return {
                'approach': approach,
                'model_name': model_name,
                'model_short': model_short,
                'run_idx': run_idx,
                'success': False,
                'training_time': training_time,
                'weights_dir': weights_dir,
                'error': result.stderr[-500:] if result.stderr else 'Unknown error'
            }

    except subprocess.TimeoutExpired:
        return {'approach': approach, 'model_name': model_name, 'model_short': model_short,
                'run_idx': run_idx, 'success': False, 'weights_dir': weights_dir, 'error': 'timeout'}
    except Exception as e:
        return {'approach': approach, 'model_name': model_name, 'model_short': model_short,
                'run_idx': run_idx, 'success': False, 'weights_dir': weights_dir, 'error': str(e)}
    finally:
        if keep_checkpoints == 'none' and os.path.exists(weights_dir):
            shutil.rmtree(weights_dir, ignore_errors=True)


def aggregate_runs(run_results: List[Dict]) -> Dict:
    """Aggregate multiple run results into mean ± std statistics."""
    if not run_results:
        return {'success': False, 'error': 'no runs'}

    maes = [r['test_mae'] for r in run_results
            if r.get('success') and r.get('test_mae') is not None
            and not (isinstance(r['test_mae'], float) and math.isnan(r['test_mae']))]
    times = [r['training_time'] for r in run_results
             if r.get('success') and r.get('training_time') is not None]

    # Use metadata from first run as representative
    base = {k: v for k, v in run_results[0].items()
            if k not in ('run_idx', 'test_mae', 'training_time', 'weights_dir')}
    base['all_runs'] = run_results
    base['n_successful_runs'] = len(maes)
    base['n_total_runs'] = len(run_results)

    if maes:
        mean_mae = sum(maes) / len(maes)
        variance = sum((x - mean_mae) ** 2 for x in maes) / len(maes)
        std_mae = math.sqrt(variance) if len(maes) > 1 else 0.0
        base.update({
            'success': True,
            'test_mae': mean_mae,   # keeps compatibility with downstream display functions
            'mean_mae': mean_mae,
            'std_mae': std_mae,
            'best_mae': min(maes),
            'worst_mae': max(maes),
            'training_time': sum(times) / len(times) if times else 0.0,
        })
    else:
        base.update({'success': False, 'error': f"All {len(run_results)} runs failed"})

    return base


def run_experiment(approach: str, model_name: str, epochs: int, batch_size: int,
                   timestamp: str, mixed_precision: bool = False, num_runs: int = 3,
                   keep_checkpoints: str = 'none') -> Dict:
    """Run num_runs training+testing experiments and aggregate (mean ± std MAE)."""
    learning_rate = get_scaled_learning_rate(model_name, batch_size)
    model_short = model_name.split('.')[0]
    print(f"  Running {approach} + {model_short} (LR: {learning_rate:.0e}, {num_runs} runs)")

    MAX_RETRIES = 3

    all_runs = []
    for run_idx in range(num_runs):
        print(f"    Run {run_idx + 1}/{num_runs} (seed={run_idx})...", end=" ", flush=True)
        run_result = None
        for attempt in range(1 + MAX_RETRIES):
            run_result = _run_single(approach, model_name, epochs, batch_size,
                                      timestamp, run_idx, mixed_precision, keep_checkpoints)
            if run_result['success']:
                break
            if attempt < MAX_RETRIES:
                print(f"FAILED (attempt {attempt + 1}/{MAX_RETRIES + 1}), retrying same seed...", end=" ", flush=True)
        all_runs.append(run_result)
        if run_result['success']:
            print(f"OK ({run_result['training_time']:.0f}s, MAE={run_result['test_mae']:.1f}°)")
        else:
            print(f"FAILED after {MAX_RETRIES + 1} attempts: {run_result.get('error', 'unknown')[:80]}")

    aggregated = aggregate_runs(all_runs)

    if keep_checkpoints == 'best' and aggregated.get('success'):
        best_run = min(
            (r for r in all_runs if r.get('success') and r.get('test_mae') is not None),
            key=lambda r: r['test_mae'],
            default=None
        )
        for r in all_runs:
            if r is not best_run and r.get('weights_dir') and os.path.exists(r['weights_dir']):
                shutil.rmtree(r['weights_dir'], ignore_errors=True)
        if best_run:
            aggregated['best_weights_dir'] = best_run.get('weights_dir')

    return aggregated

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
    """Print results matrix with mean±std MAE scores."""
    content = []
    content.append(f"\nRESULTS MATRIX (Test MAE: mean±std across runs, degrees)")
    content.append("=" * 110)

    # Create results lookup: (approach, model) -> aggregated result dict
    results_lookup = {}
    for r in results:
        if r['success'] and r.get('mean_mae') is not None:
            results_lookup[(r['approach'], r['model_name'])] = r

    # Print header — wider columns to fit "12.3°±1.4"
    model_shorts = [m.split('.')[0][:15] for m in models]
    header = f"{'Approach':<14}"
    for model_short in model_shorts:
        header += f" | {model_short:>12}"
    content.append(header)
    content.append("-" * len(header))

    # Print matrix rows
    for approach in approaches:
        row = f"{approach:<14}"
        for model in models:
            r = results_lookup.get((approach, model))
            if r is not None:
                cell = f"{r['mean_mae']:.1f}°±{r['std_mae']:.1f}"
                row += f" | {cell:>12}"
            else:
                row += f" | {'FAIL':>12}"
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
    
    def mae_str(r):
        mean = r.get('mean_mae', r.get('test_mae', 0))
        std = r.get('std_mae', 0)
        runs = r.get('n_successful_runs', 1)
        return f"{mean:.1f}°±{std:.1f} ({runs} runs)"

    # Overall best (by mean MAE)
    best_overall = min(successful, key=lambda x: x.get('mean_mae', x.get('test_mae', 999)))
    content.append(f"Overall Best: {best_overall['approach'].upper()} + {best_overall['model_short']}")
    content.append(f"  Test MAE: {mae_str(best_overall)}, Training Time: {best_overall['training_time']:.0f}s")

    # Fastest
    fastest = min(successful, key=lambda x: x['training_time'])
    content.append(f"Fastest: {fastest['approach'].upper()} + {fastest['model_short']}")
    content.append(f"  Test MAE: {mae_str(fastest)}, Training Time: {fastest['training_time']:.0f}s")

    # Best per approach
    content.append(f"\nBest Model per Approach:")
    approaches = set(r['approach'] for r in successful)
    for approach in sorted(approaches):
        approach_results = [r for r in successful if r['approach'] == approach]
        if approach_results:
            best = min(approach_results, key=lambda x: x.get('mean_mae', x.get('test_mae', 999)))
            content.append(f"  {approach:<14}: {best['model_short']} ({mae_str(best)})")

    # Best per model
    content.append(f"\nBest Approach per Model:")
    models = set(r['model_name'] for r in successful)
    for model in sorted(models):
        model_results = [r for r in successful if r['model_name'] == model]
        if model_results:
            best = min(model_results, key=lambda x: x.get('mean_mae', x.get('test_mae', 999)))
            model_short = model.split('.')[0]
            content.append(f"  {model_short:<20}: {best['approach']} ({mae_str(best)})")
    
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
        mean = result.get('mean_mae', result.get('test_mae'))
        std = result.get('std_mae', 0.0)
        mae_str = f"{mean:.1f}°±{std:.1f}" if mean is not None else "N/A"
        content.append(f"{i:<4} {result['approach']:<14} {result['model_short']:<20} {result['training_time']:<5.0f}s {mae_str:<12}")
    
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
        content.append(f"Average Training Time per Cell: {sum(training_times)/len(training_times):.0f}s")
        total_run_time = sum(r.get('n_successful_runs', 1) * r['training_time'] for r in successful)
        content.append(f"Total Training Time (all runs): {total_run_time/3600:.1f} hours")

        mean_maes = [r.get('mean_mae', r.get('test_mae')) for r in successful
                     if r.get('mean_mae') or r.get('test_mae')]
        if mean_maes:
            content.append(f"Best Mean Test MAE: {min(mean_maes):.1f}°")
            content.append(f"Average Mean Test MAE: {sum(mean_maes)/len(mean_maes):.1f}°")
    
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
    
    # Create comparison directory for output files
    os.makedirs("comparison", exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Unified comparison of approaches and models")
    parser.add_argument("--approaches", nargs='+', 
                       help="Approaches to test. Use base approaches like 'direct_angle' or combinations like 'direct_angle_mae'. (default: all base approaches)")
    parser.add_argument("--models", nargs='+', 
                       help="Models to test (default: all from architectures.py)")
    parser.add_argument("--epochs", type=int, default=1000, 
                       help="Epochs per experiment")
    parser.add_argument("--batch-size", type=int, default=16, 
                       help="Batch size for training")
    parser.add_argument("--output", type=str, default=f"{timestamp}_comparison_results.json", 
                       help="Output JSON file")
    parser.add_argument("--mixed-precision", action="store_true",
                       help="Enable mixed precision training (16-bit) for faster training and lower memory usage")
    parser.add_argument("--num-runs", type=int, default=3,
                       help="Number of training runs per cell (different random seeds). "
                            "3 = minimum for error bars, 5 = recommended for reliable conclusions. "
                            "Results reported as mean±std MAE across runs.")
    parser.add_argument("--keep-checkpoints", choices=["none", "best", "all"], default="none",
                       help="Which checkpoints to keep after each run. "
                            "'none' (default): delete all after test, minimal disk use. "
                            "'best': keep only best-MAE run per cell. "
                            "'all': keep everything.")

    args = parser.parse_args()

    # Determine approaches and models to test
    approaches = args.approaches or BASE_APPROACHES
    models = args.models or get_enabled_architectures()

    total_cells = len(approaches) * len(models)
    total_runs = total_cells * args.num_runs

    print("UNIFIED COMPARISON EXPERIMENT")
    print("=" * 50)
    print(f"Approaches: {', '.join(approaches)}")
    print(f"Models: {len(models)} models")
    print(f"Cells: {total_cells}  |  Runs per cell: {args.num_runs}  |  Total runs: {total_runs}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Mixed Precision: {'Enabled' if args.mixed_precision else 'Disabled'}")
    print(f"Checkpoint policy: {args.keep_checkpoints}  "
          f"({'~minimal disk' if args.keep_checkpoints == 'none' else 'keeping checkpoints'})")
    print()

    # Run all experiments
    results = []
    experiment_count = 0

    for approach in approaches:
        for model in models:
            experiment_count += 1
            print(f"Experiment {experiment_count}/{total_cells}")

            result = run_experiment(approach, model, args.epochs, args.batch_size,
                                    timestamp, args.mixed_precision, args.num_runs,
                                    keep_checkpoints=args.keep_checkpoints)
            results.append(result)

            # Save after every cell so a crash doesn't lose accumulated results
            with open(f"comparison/{args.output}", 'w') as f:
                json.dump(results, f, indent=2)

            # Brief status
            if result['success']:
                mean = result.get('mean_mae', result.get('test_mae', 0))
                std = result.get('std_mae', 0)
                runs_ok = result.get('n_successful_runs', 1)
                print(f"  ✓ {runs_ok}/{args.num_runs} runs OK  MAE: {mean:.1f}°±{std:.1f}  ({result['training_time']:.0f}s avg)")
            else:
                print(f"  ✗ Failed: {result.get('error', 'unknown')}")

            time.sleep(1)  # Brief pause
    
    # Print beautiful overview and save to file
    print("\n" + "="*100)
    print("COMPARISON RESULTS OVERVIEW")
    print("="*100)
    
    # Create overview file
    overview_filename = f"comparison/{timestamp}_comparison_overview.txt"
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
    
    print(f"\nDetailed results saved to: comparison/{args.output}")
    print(f"Overview saved to: {overview_filename}")

if __name__ == "__main__":
    main()