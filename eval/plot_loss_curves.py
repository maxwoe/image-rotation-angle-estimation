#!/usr/bin/env python3
"""
Training-curve diagnostics for CLS instability (Array revision, R1.4/R4.3).

Reads TensorBoard event files from the diagnostic runs (data/saved_models_diag)
and plots validation MAE over epochs for a failing plain-CE configuration
(CLS + MambaOut Tiny), a succeeding one (CLS + EfficientViT-B3), and the CSL
variant on the failing backbone. Expected picture: the failing run plateaus at
a high MAE while CSL on the same backbone converges.

Usage (from repo root, after the diagnostic runs in run_drcd_minis.sh):
    python eval/plot_loss_curves.py [--root data/saved_models_diag]
        [--output eval/results/loss_curves.pdf]
"""

import argparse
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# (root, glob pattern, label, color, linestyle) — the CE/CSL pair on
# EfficientNetV2-RW-T uses the same backbone AND the same seed; only the loss differs.
RUNS = [
    ("data/saved_models_diag2", "classification-cross_entropy-efficientnetv2_rw_t*", "CE + EffNetV2-RW-T (failing run)", "#c0392b", "-"),
    ("data/saved_models_diag2", "classification-csl-efficientnetv2_rw_t*", "CSL + EffNetV2-RW-T (same seed)", "#1e8449", "--"),
    ("data/saved_models_diag", "classification-cross_entropy-efficientvit_b3*", "CE + EfficientViT-B3 (stable)", "#2c3e8c", "-"),
]


def load_scalar(logdir, tag):
    acc = EventAccumulator(logdir, size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return None
    events = acc.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def find_run_dir(root, pattern):
    matches = sorted(glob.glob(os.path.join(root, pattern, "version_*")))
    return matches[-1] if matches else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/saved_models_diag")
    ap.add_argument("--tag", default="val_mae_deg")
    ap.add_argument("--output", default="eval/results/loss_curves.pdf")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for root, pattern, label, color, ls in RUNS:
        run_dir = find_run_dir(root, pattern)
        if run_dir is None:
            print(f"WARNING: no run found for {pattern} under {args.root}")
            continue
        data = load_scalar(run_dir, args.tag)
        if data is None:
            print(f"WARNING: tag {args.tag} not in {run_dir}")
            continue
        steps, values = data
        print(f"{label}: {len(values)} points, final {values[-1]:.2f}, min {min(values):.2f}  ({run_dir})")
        ax.plot(steps, values, label=label, color=color, linestyle=ls, linewidth=1.2)

    ax.set_xlabel("Training step", fontsize=9)
    ax.set_ylabel("Validation MAE (degrees)", fontsize=9)
    ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.5)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
