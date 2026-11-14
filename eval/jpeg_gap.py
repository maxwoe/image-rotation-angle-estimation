#!/usr/bin/env python3
"""
Clean-versus-recompressed evaluation ("JPEG gap") for a trained checkpoint.

Builds the test views as the paper's test loader does (seeded uniform angles,
rotation in the chosen rotation mode, bilinear resampling), scores the
checkpoint on every view under several conditions and reports the gap between
them. A model that estimates the angle from image content loses little when a
view is saved once as JPEG; a model that reads the rotated JPEG block grid
collapses, and its predictions snap to multiples of 90 degrees (reported as the
"axis share").

Conditions are '+'-joined operations applied to the rotated view before the
model's own preprocessing:
  clean      nothing (the protocol as published)
  png        lossless round trip (control: must equal clean)
  jpeg<Q>    one JPEG save at quality Q, e.g. jpeg90
  blur<S>    Gaussian blur with sigma S pixels, e.g. blur3
Example: --conditions clean,jpeg90,blur3+jpeg90

Images are listed alphabetically by default (--order sorted), as in all audit
results recorded before November 2026. --order pinned lists them as
train.py --test-only does (image_files.list_test_images: the pinned order of
eval/test_order/ for the paper's test sets), which pairs every image with the
same angle as the paper's evaluation and reproduces its per-seed numbers.

Example:
  python eval/jpeg_gap.py --approach cgd --ckpt data/saved_models/.../last.ckpt \
      --test-dir data/datasets/test_coco --seeds 0,1,2,3,4 \
      --conditions clean,jpeg90 --out eval/results/jpeg_gap_cgd_coco.json
"""
import argparse
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

from rotation_utils import jpeg_roundtrip, rotate_array_preserve_content  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def load_model(approach, ckpt, device):
    """Load a checkpoint for single-image inference via model.predict_angle."""
    import timm
    from architectures import get_default_input_size
    from train import get_model_class
    model = get_model_class(approach).load_from_checkpoint(ckpt, map_location=device)
    if not getattr(model, "image_size", None):
        # Training with the timm default resolution stores image_size=None; resolve
        # it the way the data loader does so predict_angle can build its transform.
        name = model.hparams.get("model_name")
        size = None
        try:
            size = timm.get_pretrained_cfg(name).input_size[1]
        except Exception:
            pass
        model.image_size = size or get_default_input_size(name)
    return model.to(device).eval()


def apply_condition(view_bgr, cond):
    for op in cond.split("+"):
        op = op.strip()
        if op == "clean":
            continue
        if op == "png":
            ok, buf = cv2.imencode(".png", view_bgr)
            view_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        elif op.startswith("jpeg"):
            view_bgr = jpeg_roundtrip(view_bgr, int(op[4:]))
        elif op.startswith("webp"):
            # a codec whose transform lattice is not the 8x8 JPEG grid (4x4 transforms)
            import io
            buf = io.BytesIO()
            Image.fromarray(cv2.cvtColor(view_bgr, cv2.COLOR_BGR2RGB)).save(buf, format="WEBP", quality=int(op[4:]))
            buf.seek(0)
            view_bgr = cv2.cvtColor(np.array(Image.open(buf).convert("RGB")), cv2.COLOR_RGB2BGR)
        elif op.startswith("blur"):
            sigma = float(op[4:])
            if sigma > 0:  # blur0 is a no-op, so condition lists can be built generically
                view_bgr = cv2.GaussianBlur(view_bgr, (0, 0), sigma)
        else:
            raise SystemExit(f"unknown condition op '{op}'")
    return view_bgr


def circular_error(pred, true):
    d = np.abs(np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)) % 360.0
    return np.minimum(d, 360.0 - d)


def axis_share(angles, tol=2.0):
    """Share of angles within tol degrees of a multiple of 90."""
    a = np.asarray(angles, dtype=float) % 90.0
    return float(np.mean(np.minimum(a, 90.0 - a) <= tol))


def summarize(pred, true):
    e = circular_error(pred, true)
    # sector / residual decomposition: is the nearest multiple of 90 deg right (the
    # quadrant, a content decision), and how precise is the angle within the sector
    signed = (np.asarray(pred, dtype=float) - np.asarray(true, dtype=float) + 180.0) % 360.0 - 180.0
    nearest = np.round(signed / 90.0) * 90.0
    return {
        "sector_acc": float((nearest == 0).mean()),
        "residual_mae": float(np.abs(signed - nearest).mean()),
        "n": int(e.size),
        "mae": float(e.mean()),
        "median": float(np.median(e)),
        "p95": float(np.percentile(e, 95)),
        "acc2": float((e <= 2).mean()),
        "acc5": float((e <= 5).mean()),
        "acc10": float((e <= 10).mean()),
        "axis_share_pred": axis_share(pred),
        "axis_share_true": axis_share(true),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--approach", required=True,
                    help="unit_vector, direct_angle, classification, cgd, psc or multibin")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test-dir", default="data/datasets/test_coco")
    ap.add_argument("--seeds", default="0,1,2,3,4", help="test rotation seeds (paper: 0-4 on COCO, 27 on DRC-D)")
    ap.add_argument("--conditions", default="clean,jpeg90")
    ap.add_argument("--rotation-mode", default="crop", choices=["crop", "crop_fixed", "circle", "pad"])
    ap.add_argument("--order", default="sorted", choices=["sorted", "pinned"],
                    help="image order that pairs images with the seeded angles: alphabetical, or as train.py --test-only")
    ap.add_argument("--limit", type=int, default=0, help="first N images only (smoke test)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=None, help="JSON with per-view predictions")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    seeds = [int(s) for s in args.seeds.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    if args.order == "pinned":

        from image_files import list_test_images

        paths = [Path(p) for p in list_test_images(args.test_dir)]

    else:

        paths = sorted(p for p in Path(args.test_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"no images in {args.test_dir}")

    print(f"model: {args.approach} from {args.ckpt} on {device}")
    print(f"test set: {len(paths)} images from {args.test_dir}, rotation mode {args.rotation_mode}")
    model = load_model(args.approach, args.ckpt, device)

    results = {"approach": args.approach, "ckpt": str(args.ckpt), "test_dir": str(args.test_dir),
               "rotation_mode": args.rotation_mode, "conditions": conditions, "seeds": seeds,
               "images": [p.name for p in paths], "per_seed": {}, "per_view": {}}
    per_cond_summaries = {c: [] for c in conditions}

    for seed in seeds:
        t0 = time.time()
        np.random.seed(seed)
        angles = np.random.uniform(0, 360, len(paths))  # as RotationDataset._generate_test_angles
        preds = {c: [] for c in conditions}
        for i, (path, angle) in enumerate(zip(paths, angles)):
            src = cv2.imread(str(path))
            if src is None:
                raise SystemExit(f"could not read {path}")
            view = rotate_array_preserve_content(src, float(angle), mode=args.rotation_mode)
            for cond in conditions:
                arr = apply_condition(view, cond)
                pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
                preds[cond].append(float(model.predict_angle(pil)))
            if (i + 1) % 200 == 0:
                print(f"  seed {seed}: {i + 1}/{len(paths)}", flush=True)
        results["per_seed"][str(seed)] = {}
        results["per_view"][str(seed)] = {"angles": angles.tolist(), **{c: preds[c] for c in conditions}}
        for cond in conditions:
            s = summarize(preds[cond], angles)
            results["per_seed"][str(seed)][cond] = s
            per_cond_summaries[cond].append(s)
            print(f"seed {seed:>2} {cond:>14}: within10 {s['acc10'] * 100:5.1f}%  within2 {s['acc2'] * 100:5.1f}%  "
                  f"MAE {s['mae']:6.2f}  median {s['median']:5.2f}  "
                  f"axis share {s['axis_share_pred'] * 100:4.1f}% (truth {s['axis_share_true'] * 100:4.1f}%)")
        print(f"seed {seed} done in {time.time() - t0:.0f} s", flush=True)

    print("\nmean over seeds")
    results["mean"] = {}
    for cond in conditions:
        keys = per_cond_summaries[cond][0].keys()
        m = {k: float(np.mean([s[k] for s in per_cond_summaries[cond]])) for k in keys}
        results["mean"][cond] = m
        print(f"{cond:>14}: within10 {m['acc10'] * 100:5.1f}%  within2 {m['acc2'] * 100:5.1f}%  MAE {m['mae']:6.2f}  "
              f"median {m['median']:5.2f}  axis share {m['axis_share_pred'] * 100:4.1f}%")
    if "clean" in conditions:
        # paired bootstrap over views (all seeds pooled) for the gap; the gate used in
        # the study is the 95% upper bound of the within-10 drop below 1 point
        brng = np.random.default_rng(0)
        true_all = np.concatenate([np.array(results["per_view"][str(s)]["angles"]) for s in seeds])
        err = {c: circular_error(np.concatenate([np.array(results["per_view"][str(s)][c]) for s in seeds]), true_all)
               for c in conditions}
        n = true_all.size
        idx = brng.integers(0, n, size=(2000, n))
        results["gap"] = {}
        for cond in conditions:
            if cond == "clean":
                continue
            d_acc = (err["clean"] <= 10).astype(float) - (err[cond] <= 10).astype(float)
            d_mae = err[cond] - err["clean"]
            acc_lo, acc_hi = np.percentile(d_acc[idx].mean(axis=1), [2.5, 97.5])
            mae_lo, mae_hi = np.percentile(d_mae[idx].mean(axis=1), [2.5, 97.5])
            results["gap"][cond] = {"acc10_drop": float(d_acc.mean()), "acc10_drop_ci95": [float(acc_lo), float(acc_hi)],
                                    "mae_increase": float(d_mae.mean()), "mae_increase_ci95": [float(mae_lo), float(mae_hi)]}
            print(f"gap clean -> {cond}: within10 {d_acc.mean() * 100:+.1f} points "
                  f"[{acc_lo * 100:+.1f}, {acc_hi * 100:+.1f}], MAE {d_mae.mean():+.2f} deg [{mae_lo:+.2f}, {mae_hi:+.2f}]")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
