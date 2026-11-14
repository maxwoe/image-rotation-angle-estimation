#!/usr/bin/env python3
"""
Harmonic spectrum of the prediction error from saved probe and cue-conflict
results: rho_k = mean cos(k (pred - truth)) for k in 1, 2, 4, 8.

rho_4 is the lattice statistic used by the gates (the JPEG grid has 90-degree
periodicity). The other harmonics are a diagnostic, not a gate: rho_1 is the
circular concentration of the error (1 = perfect), rho_2 responds to 180-degree
structure (including genuine flips, so read it next to the sector accuracy),
and rho_8 would reveal a 45-degree periodic residue that rho_4 cannot see.
Values are reported with 95% bootstrap intervals.

Reads every *_probe.json and certify/*.json (content-free textures, truth =
applied angle) and every cue_conflict/*.json (truth = content angle, and for
stamped conditions also the lattice angle phi) under eval/results/trace_audit.

  python eval/harmonics.py [--root eval/results/trace_audit] [--out eval/results/trace_audit/harmonics.tsv]
"""
import argparse
import json
from pathlib import Path

import numpy as np

KS = (1, 2, 4, 8)


def rho(pred, truth, k, rng=None, n_boot=1000):
    c = np.cos(np.deg2rad(k * (np.asarray(pred, float) - np.asarray(truth, float))))
    m = float(c.mean())
    if rng is None:
        return m, None, None
    idx = rng.integers(0, c.size, size=(n_boot, c.size))
    lo, hi = np.percentile(c[idx].mean(axis=1), [2.5, 97.5])
    return m, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("eval/results/trace_audit"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []

    files = sorted(args.root.glob("*_probe.json")) + sorted((args.root / "certify").glob("*.json"))
    for f in files:
        d = json.load(open(f))
        truth = d.get("angles")
        if truth is None:
            continue
        for cond, s in d["conditions"].items():
            if "predictions" not in s:
                continue
            row = {"file": f.relative_to(args.root).as_posix(), "kind": "probe", "condition": cond, "reference": "applied angle",
                   "n": len(truth)}
            for k in KS:
                m, lo, hi = rho(s["predictions"], truth, k, rng)
                row[f"rho{k}"] = m
                row[f"rho{k}_ci"] = (lo, hi)
            rows.append(row)

    for f in sorted((args.root / "cue_conflict").glob("*.json")):
        d = json.load(open(f))
        theta = d["theta"]
        for cond, s in d["conditions"].items():
            for ref_name, ref in (("content angle", theta), ("lattice angle", s.get("phi"))):
                if ref is None:
                    continue
                row = {"file": f.relative_to(args.root).as_posix(), "kind": "cue_conflict", "condition": cond,
                       "reference": ref_name, "n": len(theta)}
                for k in KS:
                    m, lo, hi = rho(s["predictions"], ref, k, rng)
                    row[f"rho{k}"] = m
                    row[f"rho{k}_ci"] = (lo, hi)
                rows.append(row)

    header = f"{'file':52s} {'condition':11s} {'reference':14s} {'n':>5s}" + "".join(f" {'rho' + str(k):>14s}" for k in KS)
    print(header)
    lines = ["file\tkind\tcondition\treference\tn" + "".join(f"\trho{k}\trho{k}_lo\trho{k}_hi" for k in KS)]
    for r in rows:
        cells = "".join(f" {r[f'rho{k}']:+.3f} [{r[f'rho{k}_ci'][0]:+.2f},{r[f'rho{k}_ci'][1]:+.2f}]" for k in KS)
        print(f"{r['file'][:52]:52s} {r['condition']:11s} {r['reference']:14s} {r['n']:5d}{cells}")
        lines.append("\t".join([r["file"], r["kind"], r["condition"], r["reference"], str(r["n"])]
                               + [f"{r[f'rho{k}']:.4f}\t{r[f'rho{k}_ci'][0]:.4f}\t{r[f'rho{k}_ci'][1]:.4f}" for k in KS]))
    if args.out:
        args.out.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
