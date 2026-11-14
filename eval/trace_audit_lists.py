#!/usr/bin/env python3
"""
Build checkpoint lists for eval/trace_audit.sh.

Each output line is  label|approach|checkpoint|test_dir|seeds|probe_size
(the format trace_audit.sh consumes).

  --paper            write lists for the paper's checkpoints:
                       lists/paper_drcd.txt  the best run of every head on every DRC-D backbone
                       lists/paper_coco.txt  the CGD cells on COCO 2014 and COCO 2017
  --paper-drcd       write lists/paper_drcd.txt only
  --new-since FILE  list the best checkpoint of every comparison/ run directory
                     newer than FILE (a stamp touched before a compare.py sweep),
                     written to --out

Run from the repository root.
"""
import argparse
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LISTS = ROOT / "eval/results/trace_audit/lists"

DRCD = "data/datasets/test_drcd|27|512x384"
COCO = "data/datasets/test_coco|0,1,2,3,4|640x480"

# the comparison sweeps whose best checkpoints form the six method columns of the paper's
# DRC-D table: approach, label tag, run-directory prefix
PAPER_DRCD_SWEEPS = [("classification", "csl", "20260805_193802_compare_classification_csl_"),
                     ("cgd", "cgd", "20260306_091317_compare_cgd_"),
                     ("classification", "ce", "20260306_091317_compare_classification_"),
                     ("unit_vector", "uv", "20260306_091317_compare_unit_vector_"),
                     ("psc", "psc", "20260306_091317_compare_psc_"),
                     ("direct_angle", "da", "20260306_091317_compare_direct_angle_")]

# COCO CGD cells: label, checkpoint; test set and seeds as in the paper
PAPER_COCO = [
    ("cgd_mambaout_base_coco2014", "cgd", "data/saved_models/cgd-kl_divergence-mambaout_base-32/version_0/checkpoints/"
     "cgd-kl_divergence-mambaout_base-32-version=0-epoch=0051-step=0242164-train_loss=0.0274-val_loss=0.2393-"
     "train_mae_deg_epoch=1.0981-val_mae_deg=6.1074.ckpt"),
    ("cgd_mambaout_base_coco2017", "cgd", "data/saved_models/cgd-kl_divergence-mambaout_base-32/version_1/checkpoints/"
     "cgd-kl_divergence-mambaout_base-32-version=1-epoch=0070-step=0468813-train_loss=0.0169-val_loss=0.2808-"
     "train_mae_deg_epoch=0.8398-val_mae_deg=6.2294.ckpt"),
]
# further COCO cells come from the matrix and coco2017 list files (approach|label|ckpt)
PAPER_COCO_LISTS = [("eval/matrix_models_all.txt", "coco2014"), ("eval/coco2017_all_models.txt", "coco2017")]


def best_ckpt(run_dir: Path):
    """The kept (non-last) checkpoint of a compare.py run directory, or None."""
    cks = [p for p in run_dir.rglob("*.ckpt") if p.name != "last.ckpt"]
    if not cks:
        return None
    return sorted(cks)[-1]


def paper_drcd_list():
    LISTS.mkdir(parents=True, exist_ok=True)
    lines = []
    for approach, tag, prefix in PAPER_DRCD_SWEEPS:
        for run_dir in sorted((ROOT / "comparison").glob(prefix + "*_run*")):
            ck = best_ckpt(run_dir)
            if ck is None:
                continue
            model = re.sub(r"_run\d+$", "", run_dir.name[len(prefix):])
            lines.append(f"paper_drcd_{tag}_{model}|{approach}|{ck.relative_to(ROOT)}|{DRCD}")
    (LISTS / "paper_drcd.txt").write_text("\n".join(lines) + "\n")
    print(f"paper_drcd.txt: {len(lines)} cells")


def paper_coco_list():
    LISTS.mkdir(parents=True, exist_ok=True)
    # explicit cells come from eval/eval_coco_ours.sh and get the suffix "_ours"; the
    # matrix/coco2017 list files may contain a different checkpoint of the same cell
    lines = [f"paper_{label}_ours|{approach}|{ck}|{COCO}" for label, approach, ck in PAPER_COCO if (ROOT / ck).exists()]
    for list_file, suffix in PAPER_COCO_LISTS:
        p = ROOT / list_file
        if not p.exists():
            continue
        for raw in p.read_text().splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            approach, label, ck = raw.split("|")
            if approach != "cgd" or not (ROOT / ck).exists():
                continue
            label = label if label.endswith(suffix) else f"{label}_{suffix}"
            lines.append(f"paper_cgd_{label}|cgd|{ck}|{COCO}")
    (LISTS / "paper_coco.txt").write_text("\n".join(lines) + "\n")
    print(f"paper_coco.txt: {len(lines)} cells")


def new_since(stamp: Path, out: Path, test_dir: str = "data/datasets/test_drcd"):
    t0 = stamp.stat().st_mtime
    cell = f"{test_dir}|27|512x384"
    lines = []
    for run_dir in sorted((ROOT / "comparison").glob("*_compare_*_run*")):
        if run_dir.stat().st_mtime < t0:
            continue
        ck = best_ckpt(run_dir)
        if ck is None:
            continue
        m = re.match(r"\d{8}_\d{6}_compare_(.+)_run\d+$", run_dir.name)
        name = m.group(1)
        if name.startswith("classification_csl_"):
            approach, model, tag = "classification", name[len("classification_csl_"):], "csl"
        else:
            approach, model = name.split("_", 1)
            tag = approach
        lines.append(f"{out.stem}_{tag}_{model}|{approach}|{ck.relative_to(ROOT)}|{cell}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"{out}: {len(lines)} cells")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--paper-drcd", action="store_true")
    ap.add_argument("--new-since", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--test-dir", default="data/datasets/test_drcd", help="test set for --new-since cells")
    args = ap.parse_args()
    os.chdir(ROOT)
    if args.paper or args.paper_drcd:
        paper_drcd_list()
    if args.paper:
        paper_coco_list()
    if args.new_since:
        if not args.out:
            raise SystemExit("--new-since needs --out")
        new_since(args.new_since, args.out, args.test_dir)


if __name__ == "__main__":
    main()
