"""
paper_table.py — Reproduce Table 1 from the paper.

Reads a comparison JSON (from compare.py) and outputs:
  - Human-readable left block (MAE mean±std for each approach × architecture)
  - Human-readable right block (5-run means for best approach per architecture)
  - LaTeX rows matching the paper's table format (--latex flag)

Usage:
    python paper_table.py comparison/20260306_091317_comparison_results.json
    python paper_table.py comparison/20260306_091317_comparison_results.json --latex

All 5-run means are computed from all_runs[i] entries.
std uses ddof=0 (population std), matching compare.py's aggregate_runs().
"""

import argparse
import json
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------

APPROACH_ORDER = ["direct_angle", "classification", "unit_vector", "psc", "cgd"]
APPROACH_LABELS = {
    "direct_angle": "DA",
    "classification": "CLS",
    "unit_vector": "UV",
    "psc": "PSC",
    "cgd": "CGD",
}

MODEL_ORDER = [
    "vit_tiny_patch16_224",
    "vit_base_patch16_224",
    "efficientvit_b0",
    "efficientvit_b3",
    "convnextv2_atto",
    "convnextv2_base",
    "efficientnetv2_rw_t",
    "efficientnetv2_rw_m",
    "mambaout_tiny",
    "mambaout_base",
    "focalnet_tiny_lrf",
    "focalnet_base_lrf",
    "edgenext_xx_small",
    "edgenext_base",
    "swin_tiny_patch4_window7_224",
    "swin_base_patch4_window7_224",
]

MODEL_LABELS = {
    "vit_tiny_patch16_224": "ViT-Tiny",
    "vit_base_patch16_224": "ViT-Base",
    "efficientvit_b0": "EfficientViT-B0",
    "efficientvit_b3": "EfficientViT-B3",
    "convnextv2_atto": "ConvNeXt V2 Atto",
    "convnextv2_base": "ConvNeXt V2 Base",
    "efficientnetv2_rw_t": "EfficientNetV2-RW T",
    "efficientnetv2_rw_m": "EfficientNetV2-RW M",
    "mambaout_tiny": "MambaOut Tiny",
    "mambaout_base": "MambaOut Base",
    "focalnet_tiny_lrf": "FocalNet Tiny LRF",
    "focalnet_base_lrf": "FocalNet Base LRF",
    "edgenext_xx_small": "EdgeNeXt XX-Small",
    "edgenext_base": "EdgeNeXt Base",
    "swin_tiny_patch4_window7_224": "Swin Tiny",
    "swin_base_patch4_window7_224": "Swin Base",
}

# Architecture families (pairs, for visual separator)
FAMILIES = [
    ["vit_tiny_patch16_224", "vit_base_patch16_224"],
    ["efficientvit_b0", "efficientvit_b3"],
    ["convnextv2_atto", "convnextv2_base"],
    ["efficientnetv2_rw_t", "efficientnetv2_rw_m"],
    ["mambaout_tiny", "mambaout_base"],
    ["focalnet_tiny_lrf", "focalnet_base_lrf"],
    ["edgenext_xx_small", "edgenext_base"],
    ["swin_tiny_patch4_window7_224", "swin_base_patch4_window7_224"],
]

SECONDARY_KEYS = [
    "test_drcd_median_deg",
    "test_drcd_rmse_deg",
    "test_drcd_acc_2deg",
    "test_drcd_acc_5deg",
    "test_drcd_acc_10deg",
    "test_drcd_auc_2deg",
    "test_drcd_auc_5deg",
    "test_drcd_auc_10deg",
    "test_drcd_p90_deg",
    "test_drcd_p95_deg",
]

# ---------------------------------------------------------------------------
# Data loading and aggregation
# ---------------------------------------------------------------------------

def load_results(path):
    with open(path) as f:
        data = json.load(f)

    # Build lookup: (approach, model_short) -> entry
    lookup = {}
    for entry in data:
        key = (entry["approach"], entry["model_short"])
        lookup[key] = entry
    return lookup


def compute_mae_stats(entry):
    """Return (mean, std_ddof0) for MAE across all_runs."""
    maes = [r["test_mae"] for r in entry["all_runs"]]
    return float(np.mean(maes)), float(np.std(maes, ddof=0))


def compute_secondary_means(entry):
    """Return dict of 5-run means for each secondary metric."""
    result = {}
    for key in SECONDARY_KEYS:
        vals = [r[key] for r in entry["all_runs"]]
        result[key] = float(np.mean(vals))
    return result


# ---------------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------------

def print_left_block(results):
    """Print full MAE table (left block)."""
    col_w = 14
    arch_w = 22
    header = f"{'Architecture':<{arch_w}} | " + "  ".join(
        f"{APPROACH_LABELS[a]:>{col_w}}" for a in APPROACH_ORDER
    )
    sep = "-" * len(header)
    print("\n=== Left block: MAE mean(std) across 5 runs ===")
    print(header)
    print(sep)

    # Find column bests (excluding DA for column-best tracking)
    col_best = {}
    for approach in APPROACH_ORDER:
        best_mean = None
        best_model = None
        for model in MODEL_ORDER:
            entry = results.get((approach, model))
            if entry is None:
                continue
            mean, _ = compute_mae_stats(entry)
            if best_mean is None or mean < best_mean:
                best_mean = mean
                best_model = model
        col_best[approach] = (best_model, best_mean)

    # Find overall best
    overall_best_mean = None
    overall_best_key = None
    for approach in APPROACH_ORDER:
        for model in MODEL_ORDER:
            entry = results.get((approach, model))
            if entry is None:
                continue
            mean, _ = compute_mae_stats(entry)
            if overall_best_mean is None or mean < overall_best_mean:
                overall_best_mean = mean
                overall_best_key = (approach, model)

    family_ends = {fam[-1] for fam in FAMILIES}

    for model in MODEL_ORDER:
        # Find row best (excluding DA)
        row_best_mean = None
        row_best_approach = None
        for approach in APPROACH_ORDER:
            if approach == "direct_angle":
                continue
            entry = results.get((approach, model))
            if entry is None:
                continue
            mean, _ = compute_mae_stats(entry)
            if row_best_mean is None or mean < row_best_mean:
                row_best_mean = mean
                row_best_approach = approach

        cells = []
        for approach in APPROACH_ORDER:
            entry = results.get((approach, model))
            if entry is None:
                cells.append(f"{'N/A':>{col_w}}")
                continue
            mean, std = compute_mae_stats(entry)
            cell = f"{mean:.2f}({std:.2f})"
            is_row_best = (approach == row_best_approach)
            is_col_best = (col_best[approach][0] == model)
            is_overall = ((approach, model) == overall_best_key)

            markers = ""
            if is_overall:
                cell = f"[{cell}]"   # overall best: brackets
            elif is_row_best:
                cell = f"*{cell}"    # row best: asterisk prefix
            if is_col_best:
                cell += "^"          # column best: caret suffix

            cells.append(f"{cell:>{col_w}}")

        label = MODEL_LABELS[model]
        print(f"{label:<{arch_w}} | {'  '.join(cells)}")
        if model in family_ends:
            print(sep)


def print_right_block(results):
    """Print right-block: best approach per architecture + secondary metrics."""
    arch_w = 22
    print("\n=== Right block: 5-run means for best approach per architecture ===")
    header = (
        f"{'Architecture':<{arch_w}} | "
        "Best  Med°    RMSE°   "
        "Acc@2  Acc@5  Acc@10  "
        "AUC@2  AUC@5  AUC@10  "
        "P90°    P95°"
    )
    print(header)
    print("-" * len(header))

    family_ends = {fam[-1] for fam in FAMILIES}

    for model in MODEL_ORDER:
        row_best_mean = None
        row_best_approach = None
        for approach in APPROACH_ORDER:
            if approach == "direct_angle":
                continue
            entry = results.get((approach, model))
            if entry is None:
                continue
            mean, _ = compute_mae_stats(entry)
            if row_best_mean is None or mean < row_best_mean:
                row_best_mean = mean
                row_best_approach = approach

        best_entry = results.get((row_best_approach, model))
        sec = compute_secondary_means(best_entry)
        label = MODEL_LABELS[model]
        best_label = APPROACH_LABELS[row_best_approach]
        print(
            f"{label:<{arch_w}} | "
            f"{best_label:<5} "
            f"{sec['test_drcd_median_deg']:6.2f}  "
            f"{sec['test_drcd_rmse_deg']:7.2f}  "
            f"{sec['test_drcd_acc_2deg']:5.2f}  "
            f"{sec['test_drcd_acc_5deg']:5.2f}  "
            f"{sec['test_drcd_acc_10deg']:6.2f}  "
            f"{sec['test_drcd_auc_2deg']:5.2f}  "
            f"{sec['test_drcd_auc_5deg']:5.2f}  "
            f"{sec['test_drcd_auc_10deg']:6.2f}  "
            f"{sec['test_drcd_p90_deg']:6.2f}  "
            f"{sec['test_drcd_p95_deg']:6.2f}"
        )
        if model in family_ends:
            print("-" * len(header))


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def make_latex_rows(results):
    """Return LaTeX table rows matching the paper's Table 1 format."""

    # Find column bests (min mean MAE per approach column, all 16 models)
    col_best = {}
    for approach in APPROACH_ORDER:
        best_mean = None
        best_model = None
        for model in MODEL_ORDER:
            entry = results.get((approach, model))
            if entry is None:
                continue
            mean, _ = compute_mae_stats(entry)
            if best_mean is None or mean < best_mean:
                best_mean = mean
                best_model = model
        col_best[approach] = best_model

    # Find overall best
    overall_best_mean = None
    overall_best_key = None
    for approach in APPROACH_ORDER:
        for model in MODEL_ORDER:
            entry = results.get((approach, model))
            if entry is None:
                continue
            mean, _ = compute_mae_stats(entry)
            if overall_best_mean is None or mean < overall_best_mean:
                overall_best_mean = mean
                overall_best_key = (approach, model)

    family_ends = {fam[-1] for fam in FAMILIES}
    lines = []

    for model in MODEL_ORDER:
        # Row best (excluding DA)
        row_best_mean = None
        row_best_approach = None
        for approach in APPROACH_ORDER:
            if approach == "direct_angle":
                continue
            entry = results.get((approach, model))
            if entry is None:
                continue
            mean, _ = compute_mae_stats(entry)
            if row_best_mean is None or mean < row_best_mean:
                row_best_mean = mean
                row_best_approach = approach

        # Build left-block cells
        left_cells = []
        for approach in APPROACH_ORDER:
            entry = results.get((approach, model))
            if entry is None:
                left_cells.append("--")
                continue
            mean, std = compute_mae_stats(entry)
            cell = f"{mean:.2f}({std:.2f})"
            is_row_best = (approach == row_best_approach)
            is_col_best = (col_best[approach] == model)
            is_overall = ((approach, model) == overall_best_key)

            if is_overall:
                cell = rf"\underline{{\textbf{{{cell}}}}}$^*$"
            elif is_row_best and is_col_best:
                cell = rf"\textbf{{{cell}}}$^*$"
            elif is_row_best:
                cell = rf"\textbf{{{cell}}}"
            elif is_col_best:
                cell = f"{cell}$^*$"

            left_cells.append(cell)

        # Build right-block cells
        best_entry = results.get((row_best_approach, model))
        sec = compute_secondary_means(best_entry)
        med = f"{sec['test_drcd_median_deg']:.2f}"
        rmse = f"{sec['test_drcd_rmse_deg']:.2f}"
        acc = f"{sec['test_drcd_acc_2deg']:.2f}/{sec['test_drcd_acc_5deg']:.2f}/{sec['test_drcd_acc_10deg']:.2f}"
        auc = f"{sec['test_drcd_auc_2deg']:.2f}/{sec['test_drcd_auc_5deg']:.2f}/{sec['test_drcd_auc_10deg']:.2f}"
        p = f"{sec['test_drcd_p90_deg']:.2f}/{sec['test_drcd_p95_deg']:.2f}"

        label = MODEL_LABELS[model].replace(" ", "~")
        row = (
            f"{label:<25} & "
            + " & ".join(left_cells)
            + f" & {med} & {rmse} & {acc} & {auc} & {p} \\\\"
        )
        lines.append(row)
        if model in family_ends and model != MODEL_ORDER[-1]:
            lines.append(r"\addlinespace[3pt]")

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 1 from the paper.")
    parser.add_argument("json_path", help="Path to comparison results JSON")
    parser.add_argument("--latex", action="store_true", help="Print LaTeX rows")
    args = parser.parse_args()

    results = load_results(args.json_path)

    missing = []
    for approach in APPROACH_ORDER:
        for model in MODEL_ORDER:
            if (approach, model) not in results:
                missing.append(f"{approach}/{model}")
    if missing:
        print(f"WARNING: missing entries: {missing}", file=sys.stderr)

    print_left_block(results)
    print_right_block(results)

    if args.latex:
        print("\n=== LaTeX rows ===")
        for line in make_latex_rows(results):
            print(line)


if __name__ == "__main__":
    main()
