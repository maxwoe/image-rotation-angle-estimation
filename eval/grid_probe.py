#!/usr/bin/env python3
"""
Content-free probe for angle-carrying processing traces.

Generates isotropic noise textures that contain no orientation information,
optionally saves each once as JPEG (which stamps a frame-aligned 8x8 block grid
onto it), rotates it with the paper's rotation pipeline and asks the model for
the angle. Nothing in the content indicates the rotation, so any accuracy above
chance comes from processing traces: the rotated block grid (JPEG conditions)
or the resampler residue alone (lossless condition). The grid has four-fold
symmetry, so the readable quantity is the angle modulo 90 degrees; the main
score is the share of predictions within 5 degrees of the truth modulo 90
(chance 11.1%), alongside the mean folded error (chance 22.5 degrees).

A condition written as BEFORE/AFTER applies BEFORE to the texture and AFTER to
the rotated view, e.g. jpeg95/jpeg90: a q95 grid like a COCO source photo, then
one q90 save after rotation and crop like the recompressed benchmark. A model
trained with a re-encode after rotation always sees a frame-aligned grid on top,
so these realistic conditions test it on views like its benchmark inputs, where
a faint rotated source grid survives underneath. lossless/jpeg90 is the control
with the frame-aligned grid only.

The gate is an equivalence test fixed in advance: in every condition the 95%
bootstrap upper bound of rho4 = mean cos(4 (pred - truth)) must be below
--rho4-max (0.20, ORTUS AI's shipping gate) and the upper bound of the share
within 5 deg (mod 90) minus chance must be below --excess-max (5 points). Exit
status is 1 if any condition fails, so the probe can serve as a unit test for
new checkpoints. It takes about half a minute on a GPU.

Example:
  python eval/grid_probe.py --approach cgd --ckpt data/saved_models/.../last.ckpt
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

from rotation_utils import jpeg_roundtrip, rotate_array_preserve_content  # noqa: E402

CHANCE_MOD90_WITHIN5 = 10.0 / 90.0
CHANCE_MOD90_MEAN = 22.5
CHANCE_WITHIN10 = 20.0 / 360.0


def _smooth_noise(rng, w, h, sigma):
    base = rng.standard_normal((h, w)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), sigma)
    return (base - base.mean()) / (base.std() + 1e-6)


def make_texture(rng, w, h):
    """
    Isotropic colour texture, 8 bit: per-channel Gaussian-filtered white noise
    (coarse structure), a shared mid-frequency component and fine noise. Every
    filter is isotropic, so the texture carries no orientation. The mid- and
    high-frequency content is what makes a JPEG encoder leave visible blocking.
    """
    sigma_coarse = rng.uniform(3.0, 12.0)
    sigma_mid = rng.uniform(1.0, 1.6)
    mid = _smooth_noise(rng, w, h, sigma_mid)
    fine = rng.normal(0.0, 10.0, (h, w)).astype(np.float32)
    channels = []
    for _ in range(3):
        coarse = _smooth_noise(rng, w, h, sigma_coarse)
        ch = 128.0 + 35.0 * coarse + 20.0 * mid + fine
        channels.append(np.clip(ch, 0, 255).astype(np.uint8))
    return cv2.merge(channels)  # BGR


def prepare(texture, cond):
    if cond == "lossless":
        return texture
    if cond.startswith("jpeg"):
        return jpeg_roundtrip(texture, int(cond[4:]))
    raise SystemExit(f"unknown condition '{cond}' (use lossless or jpeg<Q>, optionally BEFORE/AFTER)")


def render(texture, angle, cond, rotation_mode):
    """Texture -> BEFORE -> rotate and crop -> AFTER (AFTER only for BEFORE/AFTER conditions)."""
    before, _, after = cond.partition("/")
    view = rotate_array_preserve_content(prepare(texture, before), float(angle), mode=rotation_mode)
    if after:
        if not after.startswith("jpeg"):
            raise SystemExit(f"unknown post-rotation step '{after}' (use jpeg<Q>)")
        view = jpeg_roundtrip(view, int(after[4:]))
    return view


def folded_mod90(err):
    m = np.asarray(err, dtype=float) % 90.0
    return np.minimum(m, 90.0 - m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--approach", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=400,
                    help="textures per condition (400 gives 95%% bounds of about +-0.10 on rho4 and +-3 points "
                         "on the share, tight enough for the gates)")
    ap.add_argument("--size", default="640x480", help="texture size WxH (default 640x480, like COCO)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", default="lossless,jpeg90,jpeg75,jpeg50")
    ap.add_argument("--rotation-mode", default="crop", choices=["crop", "crop_fixed", "circle", "pad"])
    ap.add_argument("--rho4-max", type=float, default=0.20,
                    help="gate: the 95%% bootstrap upper bound of rho4 must be below this (ORTUS AI's gate: 0.20)")
    ap.add_argument("--excess-max", type=float, default=0.05,
                    help="gate: the 95%% bootstrap upper bound of (share within 5 deg mod 90 - chance) must be below this")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    w, h = (int(v) for v in args.size.lower().split("x"))
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    from jpeg_gap import load_model
    model = load_model(args.approach, args.ckpt, device)
    print(f"model: {args.approach} from {args.ckpt} on {device}")
    print(f"{args.n} isotropic textures of {w}x{h}, rotation mode {args.rotation_mode}, seed {args.seed}\n")

    rng = np.random.default_rng(args.seed)
    textures = [make_texture(rng, w, h) for _ in range(args.n)]
    angles = rng.uniform(0, 360, args.n)

    report = {"approach": args.approach, "ckpt": str(args.ckpt), "n": args.n, "seed": args.seed,
              "conditions": {}, "chance": {"mod90_within5": CHANCE_MOD90_WITHIN5,
                                           "mod90_mean": CHANCE_MOD90_MEAN, "within10": CHANCE_WITHIN10}}
    failed = []
    print(f"{'condition':>10} {'mod90 within 5':>15} {'mod90 mean':>11} {'within 10':>10}   verdict")
    print(f"{'chance':>10} {CHANCE_MOD90_WITHIN5 * 100:14.1f}% {CHANCE_MOD90_MEAN:10.1f}  {CHANCE_WITHIN10 * 100:9.1f}%")
    for cond in conditions:
        preds = []
        for tex, angle in zip(textures, angles):
            view = render(tex, angle, cond, args.rotation_mode)
            pil = Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
            preds.append(float(model.predict_angle(pil)))
        preds = np.asarray(preds)
        err = np.abs(preds - angles) % 360.0
        err = np.minimum(err, 360.0 - err)
        m = folded_mod90(err)
        # lattice statistic of ORTUS AI's audit: mean cos(4 (pred - truth)), 0 = ignores the
        # lattice, 1 = locked onto it modulo 90 deg; their shipping gate is <= 0.20
        cos4 = np.cos(np.deg2rad(4.0 * (preds - angles)))
        hit5 = (m <= 5).astype(float)
        rho4 = float(cos4.mean())
        # equivalence-style gate: 95% bootstrap upper bounds must lie below the tolerances,
        # so a noisy probe cannot pass through lack of power
        brng = np.random.default_rng(args.seed + 1)
        idx = brng.integers(0, len(preds), size=(args.bootstrap, len(preds)))
        rho4_hi = float(np.percentile(cos4[idx].mean(axis=1), 97.5))
        excess = float(hit5.mean() - CHANCE_MOD90_WITHIN5)
        excess_hi = float(np.percentile(hit5[idx].mean(axis=1), 97.5) - CHANCE_MOD90_WITHIN5)
        stats = {"mod90_within5": float(hit5.mean()), "mod90_mean": float(m.mean()),
                 "within10": float((err <= 10).mean()), "rho4": rho4, "rho4_ci95_upper": rho4_hi,
                 "excess_within5": excess, "excess_within5_ci95_upper": excess_hi,
                 "predictions": preds.tolist()}
        report["conditions"][cond] = stats
        bad = rho4_hi > args.rho4_max or excess_hi > args.excess_max
        if bad:
            failed.append(cond)
        source = "resampler residue" if cond == "lossless" else "JPEG block grid"
        verdict = f"READS {source} (gate failed)" if bad else "passes"
        print(f"{cond:>10} {stats['mod90_within5'] * 100:14.1f}% {stats['mod90_mean']:10.1f}  "
              f"{stats['within10'] * 100:9.1f}%  rho4 {rho4:+.3f} (ub {rho4_hi:+.3f})  "
              f"excess {excess * 100:+.1f} pts (ub {excess_hi * 100:+.1f})   {verdict}")
    report["angles"] = angles.tolist()
    report["failed"] = failed

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.out}")
    if failed:
        print(f"\nFAIL: the model reads processing traces in condition(s) {', '.join(failed)}")
        sys.exit(1)
    print("\nPASS: no condition above threshold")


if __name__ == "__main__":
    main()
