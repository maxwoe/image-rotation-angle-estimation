#!/usr/bin/env python3
"""
Confidence value of a CGD checkpoint: calibration on the validation split, check on the test views.

The raw score is the probability mass that the model puts within +-10 degrees of the
predicted angle (CircularGaussianDistribution.peak_mass). Because the model is trained
on Gaussian targets with sigma = 6 degrees, this mass cannot exceed about 0.92, so it is
mapped to a confidence between 0 and 1 by a logistic curve on its logit:

    confidence = sigmoid(slope * logit(mass) + intercept)

The two parameters are fitted on the validation split of the checkpoint (the split of
its own training run, views saved as JPEG at quality 90), by minimising the log loss of
the event "the predicted angle is within 10 degrees of the true angle". The test views
are never used for the fit. They are built as in the paper (pinned image order, seeded
angles, largest crop without border) and scored re-encoded at JPEG quality 90 and
without re-encoding.

As a check on images that have no orientation at all, the confidence is also computed
on the content-free noise textures of eval/grid_probe.py, where it should be low.

The result file holds the fitted parameters, ranking and calibration measures, the
effect of confidence thresholds on the test views, the confidence on the content-free
textures and, per test view, the error and the raw mass.

Example:
  python eval/confidence_calibration.py --ckpt weights/cgd_convnextv2_base_coco2017.ckpt \
      --out eval/results/confidence/cgd_convnextv2_base_coco2017.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import timm.data
import torch
from PIL import Image
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from grid_probe import make_texture, render  # noqa: E402
from image_files import list_image_files, list_test_images  # noqa: E402
from jpeg_gap import load_model  # noqa: E402
from rotation_utils import jpeg_roundtrip, rotate_array_preserve_content  # noqa: E402

TOLERANCE = 10.0  # degrees: a prediction counts as correct up to this error
THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.98, 0.99)
RELIABILITY_EDGES = (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0)


class Views(torch.utils.data.Dataset):
    """Rotated views of upright images, optionally saved once as JPEG, in the model's input form."""

    def __init__(self, paths, angles, quality, transform):
        self.paths, self.angles, self.quality, self.transform = paths, angles, quality, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        view = rotate_array_preserve_content(cv2.imread(str(self.paths[i])), float(self.angles[i]), mode="crop")
        if self.quality is not None:
            view = jpeg_roundtrip(view, self.quality)
        return self.transform(Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)))


def predict(model, transform, paths, angles, quality, workers):
    """Predicted angle and raw peak mass for every view."""
    loader = torch.utils.data.DataLoader(Views(paths, angles, quality, transform), batch_size=64, num_workers=workers)
    pred, mass = [], []
    with torch.no_grad():
        for x in loader:
            dist = model(x.to(model.device)).float()
            pred.append(model.cgd.distribution_to_angle(dist, method=model.inference_method).cpu().numpy())
            mass.append(model.cgd.peak_mass(dist, window_degrees=TOLERANCE).cpu().numpy())
    pred, mass = np.concatenate(pred).astype(np.float64), np.concatenate(mass).astype(np.float64)
    error = np.abs((pred - np.asarray(angles) + 180.0) % 360.0 - 180.0)
    return error, mass


def content_free(model, transform, n, slope, intercept):
    """Confidence on rotated noise textures without any orientation (as in eval/grid_probe.py)."""
    rng = np.random.default_rng(0)
    textures = [make_texture(rng, 640, 480) for _ in range(n)]
    angles = rng.uniform(0, 360, n)
    out = {"n": n}
    for cond in ("lossless", "jpeg90"):
        views = [render(t, a, cond, "crop") for t, a in zip(textures, angles)]
        x = torch.stack([transform(Image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))) for v in views])
        with torch.no_grad():
            dist = torch.cat([model(x[i:i + 64].to(model.device)).float() for i in range(0, n, 64)])
        conf = calibrated(model.cgd.peak_mass(dist, window_degrees=TOLERANCE).cpu().numpy().astype(np.float64), slope, intercept)
        out[cond] = {"median_confidence": float(np.median(conf)), "mean_confidence": float(conf.mean()),
                     **{f"share_with_confidence_at_least_{t}": float((conf >= t).mean()) for t in THRESHOLDS}}
    return out


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibrated(mass, slope, intercept):
    return 1.0 / (1.0 + np.exp(-(slope * logit(mass) + intercept)))


def log_loss(q, y):
    q = np.clip(q, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


def fit(mass, correct):
    y = correct.astype(np.float64)
    res = minimize(lambda w: log_loss(calibrated(mass, w[0], w[1]), y), x0=np.array([1.0, 0.0]), method="BFGS")
    return float(res.x[0]), float(res.x[1])


def auroc(score, positive):
    """Probability that a correct prediction gets a higher score than a wrong one."""
    ranks = rankdata(score)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def calibration_error(q, y, bins=15):
    """Expected calibration error over bins of equal size (most scores sit near the top)."""
    order = np.argsort(q, kind="stable")
    return float(sum(len(part) / len(q) * abs(q[part].mean() - y[part].mean()) for part in np.array_split(order, bins)))


def describe(error, mass, slope, intercept):
    correct = error <= TOLERANCE
    conf = calibrated(mass, slope, intercept)
    y = correct.astype(np.float64)
    out = {
        "n": int(len(error)),
        "mae": float(error.mean()),
        "within10": float(correct.mean()),
        "wrong": int((~correct).sum()),
        "auroc": auroc(mass, correct),
        "calibration_error_raw_mass": calibration_error(mass, y),
        "calibration_error_confidence": calibration_error(conf, y),
        "log_loss_confidence": log_loss(conf, y),
        "thresholds": [],
        "reliability": [],
    }
    for t in THRESHOLDS:
        keep = conf >= t
        out["thresholds"].append({
            "confidence_at_least": t,
            "answered": float(keep.mean()),
            "within10_among_answered": float(correct[keep].mean()),
            "mae_among_answered": float(error[keep].mean()),
            "wrong_kept": int((~correct & keep).sum()),
            "wrong_rejected": int((~correct & ~keep).sum()),
        })
    for lo, hi in zip(RELIABILITY_EDGES[:-1], RELIABILITY_EDGES[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if m.any():
            out["reliability"].append({"confidence_from": lo, "confidence_to": hi, "n": int(m.sum()),
                                       "mean_confidence": float(conf[m].mean()), "within10": float(correct[m].mean())})
    large = error > 45.0
    out["large_errors"] = {"n": int(large.sum()),
                           "share_with_confidence_at_least_0.95": float((conf[large] >= 0.95).mean()) if large.any() else None,
                           "share_with_confidence_at_least_0.98": float((conf[large] >= 0.98).mean()) if large.any() else None}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="CGD checkpoint")
    ap.add_argument("--train-dir", default=None, help="training images (default: the directory stored in the checkpoint)")
    ap.add_argument("--test-dir", default="data/datasets/test_coco")
    ap.add_argument("--seeds", default="0,1,2,3,4", help="test rotation seeds (paper: 0-4 on COCO)")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality of the re-encoded views")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--probe-n", type=int, default=400, help="number of content-free textures")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("cgd", args.ckpt, device)
    hp = model.hparams
    cfg = timm.data.resolve_model_data_config(hp.model_name)
    cfg["crop_pct"] = 1.0
    cfg["input_size"] = (3, model.image_size, model.image_size)
    transform = timm.data.create_transform(**cfg, is_training=False)

    # validation split and angles exactly as in training (RotationDataset.create_datasets)
    train_dir = args.train_dir or hp.train_dir
    _, val_paths = train_test_split(list_image_files(train_dir), test_size=hp.validation_split, random_state=hp.random_seed)
    random.seed(hp.random_seed)
    val_angles = [random.uniform(0, 360) for _ in val_paths]
    print(f"validation: {len(val_paths)} images from {train_dir}", flush=True)
    val_error, val_mass = predict(model, transform, val_paths, val_angles, args.quality, args.workers)
    slope, intercept = fit(val_mass, val_error <= TOLERANCE)
    print(f"fit: slope {slope:.4f}, intercept {intercept:.4f}", flush=True)

    # test views in the paper's pairing of images and angles
    test_paths = list_test_images(args.test_dir)
    seeds = [int(s) for s in args.seeds.split(",")]
    conditions = {"reencoded": args.quality, "not_reencoded": None}
    test = {c: {"error": [], "mass": []} for c in conditions}
    for seed in seeds:
        np.random.seed(seed)
        angles = np.random.uniform(0, 360, len(test_paths))
        for cond, quality in conditions.items():
            error, mass = predict(model, transform, test_paths, angles, quality, args.workers)
            test[cond]["error"].append(error)
            test[cond]["mass"].append(mass)
        print(f"test seed {seed} done", flush=True)

    result = {
        "checkpoint": Path(args.ckpt).name,
        "definition": "confidence = sigmoid(slope * logit(mass) + intercept); mass = probability within +-10 degrees of the predicted angle",
        "window_degrees": TOLERANCE,
        "slope": slope,
        "intercept": intercept,
        "fitted_on": {"split": "validation split of the training run", "train_dir": str(train_dir),
                      "validation_split": hp.validation_split, "random_seed": hp.random_seed, "jpeg_quality": args.quality},
        "test_dir": args.test_dir,
        "test_seeds": seeds,
        "validation": describe(val_error, val_mass, slope, intercept),
        "test": {},
        "content_free": content_free(model, transform, args.probe_n, slope, intercept),
        "per_view": {},
    }
    for cond in ("lossless", "jpeg90"):
        d = result["content_free"][cond]
        print(f"content-free textures, {cond}: median confidence {d['median_confidence']:.3f}, "
              f"at least 0.9: {100 * d['share_with_confidence_at_least_0.9']:.1f}%, "
              f"at least 0.98: {100 * d['share_with_confidence_at_least_0.98']:.1f}%")
    for cond in conditions:
        error, mass = np.concatenate(test[cond]["error"]), np.concatenate(test[cond]["mass"])
        result["test"][cond] = describe(error, mass, slope, intercept)
        result["per_view"][cond] = {"error": np.round(error, 3).tolist(), "mass": np.round(mass, 5).tolist()}
        d = result["test"][cond]
        print(f"\ntest, {cond}: MAE {d['mae']:.2f}, within 10: {100 * d['within10']:.1f}%, AUROC {d['auroc']:.3f}, "
              f"calibration error {d['calibration_error_raw_mass']:.3f} -> {d['calibration_error_confidence']:.3f}")
        for row in d["thresholds"]:
            print(f"  confidence >= {row['confidence_at_least']:.2f}: answered {100 * row['answered']:5.1f}%, "
                  f"within 10: {100 * row['within10_among_answered']:5.1f}%, MAE {row['mae_among_answered']:.2f}, "
                  f"wrong rejected {row['wrong_rejected']} of {d['wrong']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
