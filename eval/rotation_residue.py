#!/usr/bin/env python3
"""
Rotation-residue control: does a model read the faint interpolation pattern
that the rotation itself leaves in the image?

Rotating an image interpolates every output pixel from the source pixel
lattice. The residue of that interpolation is aligned with the source lattice,
so it turns with the content and carries the angle modulo 90 degrees, like the
JPEG grid but far weaker. The test pipeline rotates with bilinear interpolation
at the image's own resolution. A supersampled rotation leaves almost no such
residue: the source is upsampled by --factor (bicubic), rotated and cropped at
that resolution, and averaged back down (area resampling), which integrates the
residue away while keeping the content.

Two parts, both ending with one JPEG save at --quality after rotation and crop:

  photos    every COCO test view (sorted order, test seeds) rotated normally and
            supersampled; R = within-2 accuracy (normal) - within-2 accuracy
            (supersampled) is the precision drawn from the residue. Rule fixed
            in advance: the 95% paired-bootstrap upper bound of R (images
            resampled, seeds kept together) must be below --r-max (5 points).
  textures  the content-free probe condition lossless/jpeg90 (frame-aligned grid
            only) rendered both ways; if the share excess and rho4 of a model
            vanish with the supersampled rotation, its texture signal was the
            residue.

  python eval/rotation_residue.py --models LIST --out-dir eval/results/trace_audit/rotation_residue
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

from rotation_utils import rotate_array_preserve_content  # noqa: E402
from jpeg_gap import circular_error, load_model  # noqa: E402
from grid_probe import CHANCE_MOD90_WITHIN5, folded_mod90, make_texture  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
FORMS = ("normal", "supersampled")


def rotate(src, angle, form, factor, rotation_mode):
    if form == "normal":
        return rotate_array_preserve_content(src, float(angle), mode=rotation_mode)
    h, w = src.shape[:2]
    big = cv2.resize(src, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)
    view = rotate_array_preserve_content(big, float(angle), mode=rotation_mode)
    vh, vw = view.shape[:2]
    return cv2.resize(view, (max(1, round(vw / factor)), max(1, round(vh / factor))), interpolation=cv2.INTER_AREA)


def jpeg_bytes(view_bgr, quality):
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(view_bgr, cv2.COLOR_BGR2RGB)).save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def predict(model, blobs):
    return np.asarray([float(model.predict_angle(Image.open(io.BytesIO(b)).convert("RGB"))) for b in blobs])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--test-dir", default="data/datasets/test_coco")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--rotation-mode", default="crop")
    ap.add_argument("--textures", type=int, default=1000)
    ap.add_argument("--r-max", type=float, default=0.05, help="rule: 95%% upper bound of R must be below this")
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
    photos = {f: {s: [] for s in seeds} for f in FORMS}
    for i, p in enumerate(paths):
        src = cv2.imread(str(p))
        for s in seeds:
            for f in FORMS:
                photos[f][s].append(jpeg_bytes(rotate(src, angles[s][i], f, args.factor, args.rotation_mode), args.quality))
    trng = np.random.default_rng(0)
    textures = [make_texture(trng, 640, 480) for _ in range(args.textures)]
    t_angles = trng.uniform(0, 360, args.textures)
    tex = {f: [jpeg_bytes(rotate(t, a, f, args.factor, args.rotation_mode), args.quality)
               for t, a in zip(textures, t_angles)] for f in FORMS}
    print(f"rendered {len(paths)} images x {len(seeds)} seeds and {args.textures} textures, both forms, "
          f"in {time.time() - t0:.0f} s", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    brng = np.random.default_rng(1)
    boot_img = brng.integers(0, len(paths), size=(args.bootstrap, len(paths)))
    boot_tex = brng.integers(0, args.textures, size=(args.bootstrap, args.textures))
    for line in args.models.read_text().splitlines():
        if not line.strip():
            continue
        approach, tag, ck = line.strip().split("|")[:3]
        t1 = time.time()
        model = load_model(approach, ck, device)
        res = {"tag": tag, "approach": approach, "ckpt": ck, "n_images": len(paths), "seeds": seeds,
               "quality": args.quality, "factor": args.factor, "r_max": args.r_max, "photos": {}, "textures": {}}
        hit2 = {}
        for f in FORMS:
            e = np.stack([circular_error(predict(model, photos[f][s]), angles[s]) for s in seeds], axis=1)
            hit2[f] = (e <= 2).mean(axis=1)
            res["photos"][f] = {"acc2": float((e <= 2).mean()), "acc10": float((e <= 10).mean()),
                                "mae": float(e.mean()), "median": float(np.median(e))}
        r_img = hit2["normal"] - hit2["supersampled"]
        r_boot = r_img[boot_img].mean(axis=1)
        res.update({"R": float(r_img.mean()), "R_ci95": [float(np.percentile(r_boot, 2.5)), float(np.percentile(r_boot, 97.5))]})
        res["pass"] = res["R_ci95"][1] < args.r_max
        for f in FORMS:
            preds = predict(model, tex[f])
            err = circular_error(preds, t_angles)
            hit5 = (folded_mod90(err) <= 5).astype(float)
            cos4 = np.cos(np.deg2rad(4.0 * (preds - t_angles)))
            res["textures"][f] = {"excess_within5": float(hit5.mean() - CHANCE_MOD90_WITHIN5),
                                  "excess_within5_ci95_upper": float(np.percentile(hit5[boot_tex].mean(axis=1), 97.5)
                                                                     - CHANCE_MOD90_WITHIN5),
                                  "rho4": float(cos4.mean()),
                                  "rho4_ci95_upper": float(np.percentile(cos4[boot_tex].mean(axis=1), 97.5))}
        (args.out_dir / f"{tag}.json").write_text(json.dumps(res, indent=1))
        ph, tx = res["photos"], res["textures"]
        print(f"{tag:52s} photos within-2 normal {ph['normal']['acc2'] * 100:5.1f} super {ph['supersampled']['acc2'] * 100:5.1f} "
              f"R {res['R'] * 100:+.1f} [{res['R_ci95'][0] * 100:+.1f}, {res['R_ci95'][1] * 100:+.1f}] "
              f"{'pass' if res['pass'] else 'FAIL'} | textures excess normal {tx['normal']['excess_within5'] * 100:+.1f} "
              f"super {tx['supersampled']['excess_within5'] * 100:+.1f}, rho4 {tx['normal']['rho4']:+.3f} -> "
              f"{tx['supersampled']['rho4']:+.3f}  ({time.time() - t1:.0f} s)", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
