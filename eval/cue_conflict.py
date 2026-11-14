#!/usr/bin/env python3
"""
Cue-conflict evaluation: does a model follow the image content or the JPEG lattice?

For each test image the content is rotated by a random angle theta, while a
JPEG block lattice is stamped at a chosen angle phi = theta + delta relative to
the frame. The view is built as

    source -> (optional source-grid suppression: Gaussian blur) -> rotate by -delta
           -> JPEG save at --quality (stamps a lattice aligned with this frame)
           -> rotate by theta + delta -> centre crop to the largest rectangle inscribed
              in the source rotated by theta (the paper's crop for the content angle)

so the content ends at theta and the stamped lattice at phi = theta + delta.
delta = 0 reproduces the benchmark's situation (lattice turned with the content);
delta = -theta makes the lattice frame-aligned, as a camera produces it (written
as "camera" below); other deltas put the lattice in conflict with the content.
A "none" condition stamps no lattice (blur only), and "protocol" is the plain
benchmark view (no blur, no stamp).

Reported per condition, all relative to the content angle theta unless stated:
  within10      share of predictions within 10 deg of theta
  MAE           mean circular error against theta
  sector acc    share whose nearest multiple of 90 deg is correct (quadrant)
  residual MAE  mean error after removing the nearest multiple of 90 deg
  rho4(content) mean cos 4(pred - theta): fine-angle lock onto the content
  rho4(lattice) mean cos 4(pred - phi):   fine-angle lock onto the stamped lattice
  pull          share of predictions closer (mod 90) to the lattice than to the content

A content-driven model keeps rho4(content) high and rho4(lattice) near zero for
every delta; a lattice reader shows the reverse as soon as delta != 0.

Example:
  python eval/cue_conflict.py --approach cgd --ckpt path/to.ckpt --test-dir data/datasets/test_coco --n 300
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from rotation_utils import jpeg_roundtrip, largest_rotated_rect, rotate_image  # noqa: E402
from jpeg_gap import load_model, circular_error  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def center_crop(img, wr, hr):
    h, w = img.shape[:2]
    y1 = h // 2 - int(hr / 2)
    x1 = w // 2 - int(wr / 2)
    return img[y1:y1 + int(hr), x1:x1 + int(wr)]


def build_view(src, theta, delta, quality, suppress_sigma, stamp=True):
    """Content at theta, lattice at theta + delta (if stamp), paper crop for theta."""
    h, w = src.shape[:2]
    img = src
    if suppress_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), suppress_sigma)
    # the stamp is aligned with the frame it is applied in, so: rotate by -delta,
    # stamp (lattice at 0 in this frame), rotate by theta + delta -> content at
    # theta, lattice at theta + delta relative to the final frame
    first, _ = rotate_image(img, -delta, expand=True)
    if stamp:
        first = jpeg_roundtrip(first, quality)
    second, _ = rotate_image(first, theta + delta, expand=True)
    wr, hr = largest_rotated_rect(w, h, math.radians(theta))
    return center_crop(second, wr, hr)


def fold90(x):
    m = np.asarray(x, dtype=float) % 90.0
    return np.minimum(m, 90.0 - m)


def stats(preds, theta, phi):
    preds, theta = np.asarray(preds), np.asarray(theta)
    err = circular_error(preds, theta)
    signed = (preds - theta + 180.0) % 360.0 - 180.0
    nearest = np.round(signed / 90.0) * 90.0
    residual = np.abs(signed - nearest)
    out = {
        "within10": float((err <= 10).mean()),
        "mae": float(err.mean()),
        "sector_acc": float((nearest == 0).mean()),
        "residual_mae": float(residual.mean()),
        "rho4_content": float(np.mean(np.cos(np.deg2rad(4.0 * (preds - theta))))),
    }
    if phi is not None:
        phi = np.asarray(phi)
        out["rho4_lattice"] = float(np.mean(np.cos(np.deg2rad(4.0 * (preds - phi)))))
        out["pull"] = float(np.mean(fold90(preds - phi) < fold90(preds - theta)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--approach", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test-dir", default="data/datasets/test_coco")
    ap.add_argument("--n", type=int, default=300, help="images (first N in sorted order)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quality", type=int, default=75, help="JPEG quality of the stamped lattice")
    ap.add_argument("--suppress-sigma", type=float, default=2.5,
                    help="Gaussian blur of the source before stamping, to weaken the source's own grid")
    ap.add_argument("--deltas", default="0,-20,20,45",
                    help="lattice offsets relative to the content (deg); 'camera' (frame-aligned) is always added")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    paths = sorted(p for p in Path(args.test_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)[:args.n]
    rng = np.random.default_rng(args.seed)
    thetas = rng.uniform(0, 360, len(paths))
    model = load_model(args.approach, args.ckpt, device)
    print(f"model: {args.approach} from {args.ckpt} on {device}; {len(paths)} images, lattice q{args.quality}, "
          f"source suppression sigma {args.suppress_sigma}\n")

    conditions = [("protocol", None, False, 0.0), ("none", None, False, args.suppress_sigma)]
    for d in args.deltas.split(","):
        conditions.append((f"delta{int(d):+d}", float(d), True, args.suppress_sigma))
    conditions.append(("camera", "camera", True, args.suppress_sigma))

    report = {"approach": args.approach, "ckpt": str(args.ckpt), "n": len(paths), "quality": args.quality,
              "suppress_sigma": args.suppress_sigma, "theta": thetas.tolist(), "conditions": {}}
    print(f"{'condition':>10} {'within10':>8} {'MAE':>6} {'sector':>7} {'resid':>6} {'rho4 content':>13} "
          f"{'rho4 lattice':>13} {'pull':>5}")
    srcs = [cv2.imread(str(p)) for p in paths]
    for name, delta, stamp, sigma in conditions:
        preds, phis = [], []
        for src, theta in zip(srcs, thetas):
            if delta == "camera":
                d = -theta          # lattice at theta + d = 0, i.e. frame-aligned
            elif delta is None:
                d = 0.0
            else:
                d = delta
            view = build_view(src, float(theta), float(d), args.quality, sigma, stamp=stamp)
            pil = Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
            preds.append(float(model.predict_angle(pil)))
            phis.append((theta + d) % 360.0)
        s = stats(preds, thetas, phis if stamp else None)
        s["predictions"] = preds
        s["phi"] = phis if stamp else None
        report["conditions"][name] = s
        lat = f"{s['rho4_lattice']:+13.3f} {s['pull'] * 100:4.0f}%" if stamp else f"{'-':>13} {'-':>5}"
        print(f"{name:>10} {s['within10'] * 100:7.1f}% {s['mae']:6.2f} {s['sector_acc'] * 100:6.1f}% "
              f"{s['residual_mae']:6.2f} {s['rho4_content']:+13.3f} {lat}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
