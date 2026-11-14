#!/usr/bin/env python3
"""
Source-grid control: does a model's precision on recompressed views come from
the source photo's rotated JPEG grid surviving the final re-encode?

Renders the benchmark views twice per image (test seed as in jpeg_gap.py,
sorted file order), both ending with one JPEG save at --quality after rotation
and crop:

  standard   source -> rotate -> crop -> JPEG q          (the q90 benchmark form)
  scrubbed   source -> Lanczos downscale by --factor -> rotate -> crop -> JPEG q
             (the downscale suppresses the source's 8x8 grid before it is rotated;
             it also removes some fine detail, equally for every model)

A model that reads only content loses a little precision on the scrubbed views
(less detail). A model that reads the rotated source grid through the re-encode
loses its extra precision: within-2 and the median fall back to content-only levels.

  python eval/source_grid_control.py --model classification:path/a.ckpt --model cgd:path/b.ckpt
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from rotation_utils import jpeg_roundtrip, rotate_array_preserve_content  # noqa: E402
from jpeg_gap import load_model, summarize  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="approach:checkpoint, repeatable")
    ap.add_argument("--test-dir", default="data/datasets/test_coco")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--factor", type=float, default=2.5, help="source downscale factor for the scrubbed form")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(p for p in Path(args.test_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        paths = paths[:args.limit]
    np.random.seed(args.seed)
    angles = np.random.uniform(0, 360, len(paths))

    views = {"standard": [], "scrubbed": []}
    for p, a in zip(paths, angles):
        src = cv2.imread(str(p))
        views["standard"].append(jpeg_roundtrip(rotate_array_preserve_content(src, float(a)), args.quality))
        h, w = src.shape[:2]
        small = cv2.resize(src, (int(round(w / args.factor)), int(round(h / args.factor))),
                           interpolation=cv2.INTER_LANCZOS4)
        views["scrubbed"].append(jpeg_roundtrip(rotate_array_preserve_content(small, float(a)), args.quality))

    report = {"seed": args.seed, "quality": args.quality, "factor": args.factor, "n": len(paths), "models": {}}
    print(f"{len(paths)} views, seed {args.seed}, final JPEG q{args.quality}, source downscale x{args.factor}\n")
    print(f"{'model':40s} {'form':9s} {'within10':>8s} {'within2':>8s} {'MAE':>6s} {'median':>7s}")
    for spec in args.model:
        approach, ck = spec.split(":", 1)
        model = load_model(approach, ck, device)
        name = f"{approach}:{Path(ck).parts[-4] if len(Path(ck).parts) > 4 else Path(ck).stem}"
        report["models"][name] = {}
        for form, vs in views.items():
            preds = [float(model.predict_angle(Image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB)))) for v in vs]
            s = summarize(preds, angles)
            report["models"][name][form] = s
            print(f"{name[:40]:40s} {form:9s} {s['acc10'] * 100:7.1f}% {s['acc2'] * 100:7.1f}% "
                  f"{s['mae']:6.2f} {s['median']:7.2f}")
        del model
        torch.cuda.empty_cache()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
