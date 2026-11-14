#!/usr/bin/env python3
"""
Source-grid reliance on real photos: how much of a model's precision on the
recompressed benchmark comes from the source photo's rotated JPEG grid that
survives the final re-encode.

Every test view is rendered in three forms, with identical images, angles and
test seeds (sorted file order, as in jpeg_gap.py), all ending with one JPEG save
at --quality after rotation and crop:

  standard    source -> rotate -> crop -> JPEG q                  (the q90 benchmark view)
  suppressed  source -> blur sigma -> rotate -> crop -> JPEG q    (source grid removed, some detail lost)
  restamped   source -> blur sigma -> JPEG q95 -> rotate -> crop -> JPEG q
              (same detail as suppressed, with a fresh rotated source grid put back)

suppressed and restamped differ only in the rotated source grid, so

  G = within-2 accuracy (restamped) - within-2 accuracy (suppressed)

is the precision the model draws from a rotated source grid at matched detail.
A content-reading model has G near 0. The rule is fixed in advance: the 95%
paired-bootstrap upper bound of G (images resampled, all seeds of an image kept
together) must be below --g-max (5 points). The median error is reported
alongside.

Positive control: --quality 0 replaces the final JPEG save by a lossless one
(the published protocol). The paper checkpoints, which read the rotated grid,
must then show a large G; on q90 views they collapse in every form, so they
cannot serve as controls there.

Views are rendered once and shared by all models of a run.

  python eval/source_grid_reliance.py --models eval/results/trace_audit/lists/reliance_w1.txt \
      --out-dir eval/results/trace_audit/source_grid_reliance
  models file: one "approach|tag|checkpoint" per line
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from rotation_utils import jpeg_roundtrip, rotate_array_preserve_content  # noqa: E402
from jpeg_gap import circular_error, load_model  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
FORMS = ("standard", "suppressed", "restamped")


def encode_jpeg(view_bgr, quality):
    """The final save: JPEG at the given quality, or lossless PNG for quality 0 (the published protocol)."""
    buf = io.BytesIO()
    pil = Image.fromarray(cv2.cvtColor(view_bgr, cv2.COLOR_BGR2RGB))
    if int(quality) == 0:
        pil.save(buf, format="PNG")
    else:
        pil.save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def render(src, angle, form, sigma, quality, rotation_mode):
    if form != "standard":
        src = cv2.GaussianBlur(src, (0, 0), sigma)
        if form == "restamped":
            src = jpeg_roundtrip(src, 95)
    view = rotate_array_preserve_content(src, float(angle), mode=rotation_mode)
    return encode_jpeg(view, quality)  # the final save; the bytes are exactly what the model reads


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--test-dir", default="data/datasets/test_coco")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--quality", type=int, default=90, help="final JPEG quality; 0 = lossless PNG (positive control)")
    ap.add_argument("--sigma", type=float, default=2.5, help="source blur that suppresses the 8 px grid (residual 15%%)")
    ap.add_argument("--rotation-mode", default="crop")
    ap.add_argument("--g-max", type=float, default=0.05, help="rule: 95%% upper bound of G must be below this")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="first N images only (smoke test)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(p for p in Path(args.test_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        paths = paths[:args.limit]
    seeds = [int(s) for s in args.seeds.split(",")]
    angles = {}
    for s in seeds:
        np.random.seed(s)
        angles[s] = np.random.uniform(0, 360, len(paths))

    t0 = time.time()
    views = {f: {s: [] for s in seeds} for f in FORMS}
    for i, p in enumerate(paths):
        src = cv2.imread(str(p))
        for s in seeds:
            for f in FORMS:
                views[f][s].append(render(src, angles[s][i], f, args.sigma, args.quality, args.rotation_mode))
    print(f"rendered {len(paths)} images x {len(seeds)} seeds x {len(FORMS)} forms in {time.time() - t0:.0f} s", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    boot_idx = rng.integers(0, len(paths), size=(args.bootstrap, len(paths)))
    for line in args.models.read_text().splitlines():
        if not line.strip():
            continue
        approach, tag, ck = line.strip().split("|")[:3]
        t1 = time.time()
        model = load_model(approach, ck, device)
        res = {"tag": tag, "approach": approach, "ckpt": ck, "n_images": len(paths), "seeds": seeds,
               "quality": args.quality, "sigma": args.sigma, "g_max": args.g_max, "forms": {}}
        hit2 = {}
        for f in FORMS:
            errs = []
            for s in seeds:
                preds = [float(model.predict_angle(Image.open(io.BytesIO(b)).convert("RGB"))) for b in views[f][s]]
                errs.append(circular_error(preds, angles[s]))
            e = np.stack(errs, axis=1)  # images x seeds
            hit2[f] = (e <= 2).mean(axis=1)  # per-image share within 2 deg over seeds
            res["forms"][f] = {"acc2": float((e <= 2).mean()), "acc10": float((e <= 10).mean()),
                               "mae": float(e.mean()), "median": float(np.median(e))}
        g_img = hit2["restamped"] - hit2["suppressed"]
        g = float(g_img.mean())
        g_hi = float(np.percentile(g_img[boot_idx].mean(axis=1), 97.5))
        g_lo = float(np.percentile(g_img[boot_idx].mean(axis=1), 2.5))
        res.update({"G": g, "G_ci95": [g_lo, g_hi], "pass": g_hi < args.g_max})
        (args.out_dir / f"{tag}.json").write_text(json.dumps(res, indent=1))
        fm = res["forms"]
        print(f"{tag:52s} within-2 std {fm['standard']['acc2'] * 100:5.1f}  sup {fm['suppressed']['acc2'] * 100:5.1f}  "
              f"rest {fm['restamped']['acc2'] * 100:5.1f} | median {fm['suppressed']['median']:.2f} -> "
              f"{fm['restamped']['median']:.2f} | G {g * 100:+.1f} [{g_lo * 100:+.1f}, {g_hi * 100:+.1f}] "
              f"{'pass' if res['pass'] else 'FAIL'}  ({time.time() - t1:.0f} s)", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
