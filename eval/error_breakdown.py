#!/usr/bin/env python3
"""
Error-type breakdown near cardinal rotations (Array revision, R1.7).

Pools per-sample predictions of the best COCO model (CGD + MambaOut Base) over
five test seeds and quantifies the qualitative claim in the Analysis section
that the worst predictions cluster near cardinal rotations: errors are
classified as fine (<10°), perpendicular flips (90°±10°), polarity flips
(>=170°), and other. Also checks whether failures correlate with the
ground-truth angle being near a cardinal orientation.

Usage (from repo root):
    python eval/error_breakdown.py [--checkpoint ...] [--test-dir ...]
        [--seeds 0 1 2 3 4] [--output-json eval/results/error_breakdown.json]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.generate_qualitative_figure import load_model, run_inference


def circ_err(pred, gt):
    d = np.abs(np.asarray(pred) - np.asarray(gt)) % 360.0
    return np.minimum(d, 360.0 - d)


def dist_to_cardinal(angles):
    """Circular distance of each angle to the nearest cardinal (0/90/180/270)."""
    a = np.asarray(angles) % 90.0
    return np.minimum(a, 90.0 - a)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str,
                    default="data/saved_models/cgd-kl_divergence-mambaout_base-32/version_0/checkpoints/"
                            "cgd-kl_divergence-mambaout_base-32-version=0-epoch=0051-step=0242164-"
                            "train_loss=0.0274-val_loss=0.2393-train_mae_deg_epoch=1.0981-val_mae_deg=6.1074.ckpt")
    ap.add_argument("--test-dir", type=str, default="data/datasets/test_coco")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--output-json", type=str, default="eval/results/error_breakdown.json")
    args = ap.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)

    records = []
    for seed in args.seeds:
        print(f"Running inference with seed={seed}...")
        results = run_inference(model, args.test_dir, seed)
        for r in results:
            records.append({"seed": seed, "gt": r["angle_applied"],
                            "pred": r["angle_predicted"], "error": r["error"]})

    err = np.array([r["error"] for r in records])
    gt = np.array([r["gt"] for r in records])
    n = len(err)
    print(f"\nPooled {n} predictions over seeds {args.seeds}: "
          f"MAE {err.mean():.2f}°, median {np.median(err):.2f}°")

    # --- Error-type breakdown ---------------------------------------------
    fine = err < 10.0
    perp = np.abs(err - 90.0) <= 10.0
    flip = err >= 170.0
    other = ~(fine | perp | flip)
    fail = err >= 10.0

    print("\nERROR-TYPE BREAKDOWN")
    print(f"  fine (<10°):                {fine.sum():5d}  ({100*fine.mean():.2f}%)")
    print(f"  perpendicular (90°±10°):    {perp.sum():5d}  ({100*perp.mean():.2f}%)")
    print(f"  polarity flip (>=170°):     {flip.sum():5d}  ({100*flip.mean():.2f}%)")
    print(f"  other (10-80°, 100-170°):   {other.sum():5d}  ({100*other.mean():.2f}%)")
    if fail.sum():
        share_cardinal = (perp.sum() + flip.sum()) / fail.sum()
        print(f"  -> of all failures (>=10°): {100*share_cardinal:.1f}% are near-perpendicular "
              f"or near-flip errors")

    # --- Does the GT angle matter? ----------------------------------------
    card = dist_to_cardinal(gt)
    near = card <= 10.0
    print("\nGROUND-TRUTH ANGLE vs FAILURE RATE")
    print(f"  GT within 10° of cardinal:  fail rate {100*fail[near].mean():.2f}%  (n={near.sum()})")
    print(f"  GT elsewhere:               fail rate {100*fail[~near].mean():.2f}%  (n={(~near).sum()})")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({
            "n": int(n), "seeds": args.seeds, "mae": float(err.mean()),
            "median": float(np.median(err)),
            "breakdown": {
                "fine_lt10": int(fine.sum()), "perpendicular_90pm10": int(perp.sum()),
                "flip_ge170": int(flip.sum()), "other": int(other.sum()),
            },
            "fail_rate_gt_near_cardinal": float(fail[near].mean()),
            "fail_rate_gt_elsewhere": float(fail[~near].mean()),
            "records": records,
        }, f)
    print(f"\nSaved breakdown + per-sample records to {args.output_json}")

    # --- Histogram figure with marked bands -------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    bins = np.arange(0, 185, 5)
    ax.hist(err, bins=bins, weights=np.ones_like(err) * 100.0 / n,
            color="#2c3e8c", edgecolor="white", linewidth=0.3)
    # No shaded bands: the two error modes are quantified in the caption and Section 5.4 instead,
    # which reads more cleanly than tinting regions of a log-scale histogram.
    ax.set_xlabel("Angle estimation error (degrees)", fontsize=9)
    ax.set_ylabel("Percentage of images (%)", fontsize=9)
    ax.set_xlim(0, 180)
    ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.5)
    out_pdf = os.path.join(os.path.dirname(args.output_json), "error_breakdown_histogram.pdf")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Histogram saved to {out_pdf}")


if __name__ == "__main__":
    main()
