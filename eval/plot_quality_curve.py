#!/usr/bin/env python3
"""
Codec-quality robustness figure: Acc@10 of CGD models on the COCO test views
without re-encoding and after one save at JPEG quality 95 to 50 (and WebP q90),
for the models trained under the conventional protocol and under the trace-free
protocol on the three COCO 2014 backbones.

Reads the per-seed results written by eval/jpeg_gap.py under
eval/results/trace_audit/quality_curve/ (paper_*.json = conventional protocol,
camera_*.json = trace-free protocol). Colour encodes the protocol, the marker
the backbone, so identity never rests on colour alone.

    python eval/plot_quality_curve.py [--root eval/results/trace_audit/quality_curve]
        [--output eval/results/quality_curve.pdf]
"""
import argparse
import json
import os

import numpy as np

CONDS = ["clean", "jpeg95", "jpeg90", "jpeg80", "jpeg70", "jpeg60", "jpeg50"]
TICKS = ["none", "95", "90", "80", "70", "60", "50"]
EXTRA = "webp90"
# validated pair (light surface): CVD and normal-vision separation, lightness band, contrast
BLUE, ORANGE = "#3a56b4", "#c2410c"
PROTOCOLS = [("camera", "trace-free protocol", BLUE, "-"), ("paper", "conventional protocol", ORANGE, "--")]
BACKBONES = [("convnext", "ConvNeXt V2 Base", "o"), ("mambaout", "MambaOut Base", "s"), ("effvit", "EfficientViT-B3", "^")]


def acc10(path, cond):
    per_seed = json.load(open(path))["per_seed"]
    return 100.0 * float(np.mean([per_seed[s][cond]["acc10"] for s in per_seed]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="eval/results/trace_audit/quality_curve")
    ap.add_argument("--output", default="eval/results/quality_curve.pdf")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = np.arange(len(CONDS))
    xw = len(CONDS) + 0.6  # WebP sits apart: a different codec, not a point on the JPEG quality axis
    for prefix, _, color, ls in PROTOCOLS:
        for key, _, marker in BACKBONES:
            path = os.path.join(args.root, f"{prefix}_{key}.json")
            ys = [acc10(path, c) for c in CONDS]
            ax.plot(x, ys, color=color, linestyle=ls, linewidth=1.2, marker=marker, markersize=4,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.plot([xw], [acc10(path, EXTRA)], color=color, linestyle="none", marker=marker, markersize=4,
                    markeredgecolor="white", markeredgewidth=0.5)
            print(f"{prefix}_{key}: " + " ".join(f"{t}={v:.1f}" for t, v in zip(TICKS, ys))
                  + f" webp90={acc10(path, EXTRA):.1f}")

    ax.set_xticks(list(x) + [xw])
    ax.set_xticklabels(TICKS + ["WebP\n90"], fontsize=8)
    ax.set_xlabel("Re-encoding of the test view (JPEG quality)", fontsize=9)
    ax.set_ylabel(r"Acc@10$^\circ$ (%)", fontsize=9)
    ax.set_ylim(0, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=8, length=3)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.axvline(len(CONDS) - 0.2, color="#cfcfcf", linewidth=0.6, linestyle=":")

    # direct labels for the two protocols (text in neutral ink, the line carries the colour)
    ax.text(3.05, 91.0, PROTOCOLS[0][1], fontsize=7.5, color="#333333", ha="left", va="top")
    ax.text(3.05, 36.5, PROTOCOLS[1][1], fontsize=7.5, color="#333333", ha="left", va="center")
    # the protocols are labelled directly; the legend only decodes the backbone markers
    handles = [Line2D([0], [0], color="#555555", linestyle="none", marker=m, markersize=4, label=lab)
               for _, lab, m in BACKBONES]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.52, 0.0),
              ncol=3, handletextpad=0.0, columnspacing=0.5, borderaxespad=0.4)

    fig.tight_layout(pad=0.5)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.output)[0] + ".png", dpi=200, bbox_inches="tight")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
