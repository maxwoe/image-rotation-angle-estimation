#!/usr/bin/env python3
"""
Model complexity and training-time table (Array revision, R2.2/R2.7).

For every architecture in the benchmark: backbone parameter count, output-head
parameter counts (360-way classification head as used by CLS/CGD, and the MLP
regression head used by DA/UV/PSC), and mean training time / epochs per method
from the paper's comparison JSON.

Usage (from repo root):
    python eval/params_time_table.py [--json comparison/20260306_091317_comparison_results.json] [--latex]
"""

import argparse
import json
import os
import sys

import numpy as np
import timm
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architectures import get_enabled_architectures


def head_params(num_features):
    """Parameters of ConfigurableRegressionHead([f/2, f/4, f/8] -> 2) incl. LayerNorm."""
    dims = [num_features, num_features // 2, num_features // 4, num_features // 8, 2]
    total = 0
    for i in range(len(dims) - 1):
        total += dims[i] * dims[i + 1] + dims[i + 1]      # linear
        if i < len(dims) - 2:
            total += 2 * dims[i + 1]                       # layer norm affine
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="comparison/20260306_091317_comparison_results.json")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    with open(args.json) as f:
        cells = json.load(f)

    print(f"{'architecture':<38} {'backbone M':>10} {'cls-head M':>10} {'reg-head M':>10}")
    arch_rows = []
    for name in get_enabled_architectures():
        model = timm.create_model(name, pretrained=False, num_classes=0)
        n_backbone = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            feats = model(torch.randn(1, 3, model.default_cfg.get("input_size", (3, 224, 224))[1],
                                      model.default_cfg.get("input_size", (3, 224, 224))[2]))
        f = feats.shape[-1]
        n_cls = f * 360 + 360
        n_reg = head_params(f)
        arch_rows.append((name, n_backbone, n_cls, n_reg))
        print(f"{name:<38} {n_backbone/1e6:>10.1f} {n_cls/1e6:>10.2f} {n_reg/1e6:>10.2f}")
        if args.latex:
            short = name.split('.')[0].replace('_', '\\_')
            print(f"  LaTeX: {short} & {n_backbone/1e6:.1f} & {n_cls/1e6:.2f} & {n_reg/1e6:.2f} \\\\")

    print("\nMEAN TRAINING TIME PER METHOD (successful runs, DRC-D, single A100)")
    by_approach = {}
    for cell in cells:
        if not cell.get("success"):
            continue
        for r in cell["all_runs"]:
            if r.get("success"):
                by_approach.setdefault(cell["approach"], []).append(
                    (r["training_time"], r.get("actual_epochs")))
    for approach, rows in sorted(by_approach.items()):
        times = np.array([t for t, _ in rows])
        epochs = np.array([e for _, e in rows if e is not None])
        print(f"  {approach:<16} n={len(times):3d}  time {times.mean()/60:6.1f} ± {times.std()/60:5.1f} min"
              f"   epochs {epochs.mean():5.1f} ± {epochs.std():4.1f}")
        if args.latex:
            print(f"    LaTeX: {approach} & {times.mean()/60:.1f} $\\pm$ {times.std()/60:.1f} & "
                  f"{epochs.mean():.0f} $\\pm$ {epochs.std():.0f} \\\\")


if __name__ == "__main__":
    main()
