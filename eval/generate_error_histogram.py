"""
Generate error distribution histogram for the paper.

Runs CGD + MambaOut Base on COCO test images across 5 test seeds,
pools all errors, and produces a single-column histogram PDF.

Usage (from repo root):
    python eval/generate_error_histogram.py

Optional arguments:
    --test-dir         Test image directory (default: data/datasets/test_coco)
    --checkpoint       Path to checkpoint file
    --seeds            Test seeds (default: 0 1 2 3 4)
    --output           Output PDF path (default: figures/error_histogram.pdf)
"""

import argparse
import sys
import os
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.generate_qualitative_figure import load_model, run_inference


def main():
    parser = argparse.ArgumentParser(description="Generate error histogram for paper")
    parser.add_argument("--test-dir", type=str, default="data/datasets/test_coco",
                        help="Directory with test images")
    parser.add_argument("--checkpoint", type=str,
                        default="data/saved_models/cgd-kl_divergence-mambaout_base-32/version_0/checkpoints/"
                                "cgd-kl_divergence-mambaout_base-32-version=0-epoch=0051-step=0242164-"
                                "train_loss=0.0274-val_loss=0.2393-train_mae_deg_epoch=1.0981-val_mae_deg=6.1074.ckpt",
                        help="Path to CGD checkpoint")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Test seeds to aggregate over")
    parser.add_argument("--output", type=str, default="eval/results/error_histogram.pdf",
                        help="Output PDF path")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)

    all_errors = []
    for seed in args.seeds:
        print(f"\nRunning inference with seed={seed}...")
        results = run_inference(model, args.test_dir, seed)
        errors = [r["error"] for r in results]
        print(f"  {len(errors)} images, MAE={np.mean(errors):.2f}°")
        all_errors.extend(errors)

    all_errors = np.array(all_errors)
    print(f"\nTotal: {len(all_errors)} predictions across {len(args.seeds)} seeds")
    print(f"Overall MAE: {np.mean(all_errors):.2f}°, Median: {np.median(all_errors):.2f}°")

    # Generate histogram
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    bins = np.arange(0, 185, 5)  # 5° bins from 0 to 180
    counts, _, patches = ax.hist(all_errors, bins=bins,
                                  weights=np.ones_like(all_errors) * 100.0 / len(all_errors),
                                  color="#2c3e8c", edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Angle estimation error (degrees)", fontsize=9)
    ax.set_ylabel("Percentage of images (%)", fontsize=9)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, None)
    ax.tick_params(labelsize=8)

    fig.tight_layout(pad=0.5)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"\nHistogram saved to {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
