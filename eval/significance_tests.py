#!/usr/bin/env python3
"""
Statistical significance tests for the method comparison (Array revision, R1.3/R4.7).

Runs are paired by seed (compare.py uses run_idx as the random seed, so run k of
every configuration shares the same train/val split). We therefore use paired
tests across the five seeds:
  - paired t-test (parametric)
  - Wilcoxon signed-rank (non-parametric; with n=5 the smallest attainable
    two-sided p is 0.0625, so it can at best be suggestive)
plus a bootstrap 95% CI of the per-seed MAE for each configuration and of the
paired mean difference.

Usage:
    python eval/significance_tests.py [--json eval/results/drcd/base_grid_seeds.json ...]
        [--top 5] [--latex]
"""

import argparse
import itertools
import json

import numpy as np
from scipy import stats


def load_cells(paths):
    cells = {}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for cell in data:
            if not cell.get("success"):
                continue
            maes = {r["run_idx"]: r["test_drcd_mae_deg"] for r in cell["all_runs"]
                    if r.get("success") and r.get("test_drcd_mae_deg") is not None}
            if maes:
                cells[(cell["approach"], cell["model_short"])] = maes
    return cells


def bootstrap_ci(values, n_boot=10000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    values = np.asarray(values)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def compare_pair(name_a, maes_a, name_b, maes_b):
    seeds = sorted(set(maes_a) & set(maes_b))
    a = np.array([maes_a[s] for s in seeds])
    b = np.array([maes_b[s] for s in seeds])
    diff = a - b

    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:  # all differences zero
        w_stat, w_p = np.nan, 1.0
    d_lo, d_hi = bootstrap_ci(diff)

    print(f"\n{name_a}  vs  {name_b}  (paired over seeds {seeds})")
    print(f"  per-seed MAE A: {np.round(a, 3).tolist()}  mean {a.mean():.3f}")
    print(f"  per-seed MAE B: {np.round(b, 3).tolist()}  mean {b.mean():.3f}")
    print(f"  mean diff (A-B): {diff.mean():+.3f}°   bootstrap 95% CI [{d_lo:+.3f}, {d_hi:+.3f}]")
    print(f"  paired t-test:   t={t_stat:+.3f}  p={t_p:.4f}")
    print(f"  Wilcoxon:        W={w_stat}  p={w_p:.4f}")
    verdict = "statistically tied" if (t_p > 0.05 and w_p > 0.05) else "significant at 0.05"
    print(f"  -> {verdict}")
    return {"pair": (name_a, name_b), "mean_diff": diff.mean(),
            "ci": (d_lo, d_hi), "t_p": t_p, "w_p": w_p}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", nargs="+",
                    default=["eval/results/drcd/base_grid_seeds.json",
                             "eval/results/drcd/csl_sweep_seeds.json"],
                    help="Comparison results JSON(s); cells are merged")
    ap.add_argument("--top", type=int, default=5, help="Number of best cells to analyze")
    ap.add_argument("--latex", action="store_true", help="Emit LaTeX rows for the CI table")
    args = ap.parse_args()

    cells = load_cells(args.json)
    ranked = sorted(cells.items(), key=lambda kv: np.mean(list(kv[1].values())))

    print(f"Loaded {len(cells)} successful cells from {args.json}")
    print(f"\nTop {args.top} configurations (mean MAE over seeds, bootstrap 95% CI):")
    top = ranked[:args.top]
    for (approach, model), maes in top:
        vals = np.array(list(maes.values()))
        lo, hi = bootstrap_ci(vals)
        line = f"  {approach:>14} + {model:<20} {vals.mean():.3f}° ± {vals.std():.3f}  CI [{lo:.3f}, {hi:.3f}]"
        print(line)
        if args.latex:
            print(f"    LaTeX: {approach} & {model} & {vals.mean():.2f} & [{lo:.2f}, {hi:.2f}] \\\\")

    print("\n" + "=" * 70)
    print("PAIRWISE TESTS (all pairs among top configurations)")
    results = []
    for ((ap_a, m_a), maes_a), ((ap_b, m_b), maes_b) in itertools.combinations(top, 2):
        results.append(compare_pair(f"{ap_a}+{m_a}", maes_a, f"{ap_b}+{m_b}", maes_b))

    n_tied = sum(1 for r in results if r["t_p"] > 0.05 and r["w_p"] > 0.05)
    print(f"\nSummary: {n_tied}/{len(results)} top pairs are statistically indistinguishable "
          f"at alpha=0.05 across five paired seeds.")


if __name__ == "__main__":
    main()
