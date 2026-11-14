#!/usr/bin/env python3
"""
Re-score OAD-360 (Maji and Bose, released ViT model) on this repository's own
COCO test views, so that the prior-work row of Table 2 is on the same protocol
as our models: max-area crop, test seeds 0-4, scored on lossless (PNG) views
and on the same views saved once as JPEG q90.

The paper's earlier OAD-360 number (10.07 deg) came from eval/oad360/test_oad360.py,
which rotates with PIL (expand, white fill) and saves q95 JPEGs, i.e. a padded,
recompressed protocol of its own.

Runs inside the OAD-360 repository's environment (TensorFlow, Keras ViT):
  PYTHONPATH=<dioad repo> python eval/oad360/rescore_oad360.py --dioad-repo <path> \
      --test-dir data/datasets/test_coco --seeds 0,1,2,3,4 --conditions png,jpeg90 --out eval/results/oad360_rescore.json

--sign-check renders a handful of views and reports which angle convention
(raw, -raw) matches the applied rotation, before scoring anything.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from rotation_utils import rotate_array_preserve_content, jpeg_roundtrip  # noqa: E402
from image_files import list_test_images  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def circular_error(pred, true):
    d = np.abs(np.asarray(pred, float) - np.asarray(true, float)) % 360.0
    return np.minimum(d, 360.0 - d)


def auc_at(e, threshold, n_points=100):
    """Area under the cumulative-accuracy curve up to threshold, as in metrics.CircularMetrics.auc_at_threshold."""
    ts = np.linspace(0.0, threshold, n_points)
    return float(np.trapz([(e <= t).mean() for t in ts], ts) / threshold)


def summarize(pred, true):
    e = circular_error(pred, true)
    signed = (np.asarray(pred, float) - np.asarray(true, float) + 180.0) % 360.0 - 180.0
    nearest = np.round(signed / 90.0) * 90.0
    return {"n": int(e.size), "mae": float(e.mean()), "median": float(np.median(e)), "rmse": float(np.sqrt((e ** 2).mean())),
            "p90": float(np.percentile(e, 90)), "p95": float(np.percentile(e, 95)),
            "acc2": float((e <= 2).mean()), "acc5": float((e <= 5).mean()), "acc10": float((e <= 10).mean()),
            "auc2": auc_at(e, 2.0), "auc5": auc_at(e, 5.0), "auc10": auc_at(e, 10.0),
            "sector_acc": float((nearest == 0).mean()), "residual_mae": float(np.abs(signed - nearest).mean())}


def write_view(view_bgr, cond, path):
    if cond == "png":
        cv2.imwrite(str(path.with_suffix(".png")), view_bgr)
        return path.with_suffix(".png")
    if cond.startswith("jpeg"):
        rgb = cv2.cvtColor(view_bgr, cv2.COLOR_BGR2RGB)
        p = path.with_suffix(".jpg")
        Image.fromarray(rgb).save(p, "JPEG", quality=int(cond[4:]))
        return p
    raise SystemExit(f"unknown condition {cond}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dioad-repo", required=True, type=Path)
    ap.add_argument("--test-dir", default="data/datasets/test_coco")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--conditions", default="png,jpeg90")
    ap.add_argument("--rotation-mode", default="crop")
    ap.add_argument("--sign", choices=["raw", "neg"], default=None,
                    help="convention mapping the model output to the applied CCW angle; default: decide with --sign-check")
    ap.add_argument("--sign-check", action="store_true", help="only run the convention check on 12 views and exit")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(args.dioad_repo))
    os.chdir(args.dioad_repo)          # their config uses relative weight paths
    from infer import Inference        # noqa: E402
    inference = Inference()
    os.chdir(ROOT)

    def raw_predict(path):
        return float(inference.predict("vit", str(path), postprocess_and_save=False))

    # the same image order, and therefore the same image-angle pairs, as train.py --test-only
    paths = [Path(p) for p in list_test_images(ROOT / args.test_dir)]
    if args.limit:
        paths = paths[:args.limit]
    tmp = Path(tempfile.mkdtemp(prefix="oad360_views_"))

    if args.sign_check or args.sign is None:
        rng = np.random.default_rng(0)
        picks = paths[:12]
        angles = rng.uniform(0, 360, len(picks))
        raws = []
        for p, a in zip(picks, angles):
            v = rotate_array_preserve_content(cv2.imread(str(p)), float(a), mode=args.rotation_mode)
            raws.append(raw_predict(write_view(v, "png", tmp / f"sign_{p.stem}")))
        raws = np.array(raws)
        e_raw = circular_error(raws % 360, angles).mean()
        e_neg = circular_error((-raws) % 360, angles).mean()
        chosen = "raw" if e_raw < e_neg else "neg"
        print(f"sign check on {len(picks)} views: MAE if applied = raw: {e_raw:.1f} deg; if applied = -raw: {e_neg:.1f} deg "
              f"-> using '{chosen}'")
        if args.sign_check:
            return
        args.sign = chosen

    conv = (lambda r: r % 360.0) if args.sign == "raw" else (lambda r: (-r) % 360.0)
    seeds = [int(s) for s in args.seeds.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]
    results = {"model": "OAD-360 (vit-ang-loss)", "sign": args.sign, "rotation_mode": args.rotation_mode,
               "test_dir": args.test_dir, "seeds": seeds, "conditions": conditions, "per_seed": {}, "mean": {}}
    per_cond = {c: [] for c in conditions}
    for seed in seeds:
        t0 = time.time()
        np.random.seed(seed)
        angles = np.random.uniform(0, 360, len(paths))
        preds = {c: [] for c in conditions}
        for i, (p, a) in enumerate(zip(paths, angles)):
            v = rotate_array_preserve_content(cv2.imread(str(p)), float(a), mode=args.rotation_mode)
            for c in conditions:
                f = write_view(v, c, tmp / f"s{seed}_{c}_{p.stem}")
                preds[c].append(conv(raw_predict(f)))
                f.unlink()
            if (i + 1) % 200 == 0:
                print(f"  seed {seed}: {i + 1}/{len(paths)}", flush=True)
        results["per_seed"][str(seed)] = {}
        for c in conditions:
            s = summarize(preds[c], angles)
            results["per_seed"][str(seed)][c] = s
            per_cond[c].append(s)
            print(f"seed {seed} {c:>7}: within10 {s['acc10'] * 100:5.1f}%  MAE {s['mae']:6.2f}  median {s['median']:5.2f}  "
                  f"P95 {s['p95']:5.1f}  sector {s['sector_acc'] * 100:5.1f}%")
        print(f"seed {seed} done in {time.time() - t0:.0f} s", flush=True)
    print("\nmean over seeds")
    for c in conditions:
        m = {k: float(np.mean([s[k] for s in per_cond[c]])) for k in per_cond[c][0]}
        results["mean"][c] = m
        print(f"{c:>7}: within10 {m['acc10'] * 100:5.1f}%  within2 {m['acc2'] * 100:5.1f}%  MAE {m['mae']:6.2f}  "
              f"median {m['median']:5.2f}  P90 {m['p90']:5.1f}  P95 {m['p95']:5.1f}  sector {m['sector_acc'] * 100:5.1f}%")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
