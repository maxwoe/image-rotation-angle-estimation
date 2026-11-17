#!/usr/bin/env python3
"""
COCO rotation benchmark, protocol v1 (defined in BENCHMARK.md), for any model.

Renders the test views of the paper from the 1,030 test photos (pinned image order,
five seeded sets of angles, largest crop without border), in two forms: saved once as
JPEG at quality 90 after rotation (the primary form) and not re-encoded. Then it runs
the content-free probe for processing traces on 1,000 noise textures in seven
conditions. The model is called once per image and returns an angle, so the same
script scores every model on the same pixels.

A model is connected by an adapter, a Python file with

    INFO = {"name": ..., "parameters": ..., "training_data": ..., "input": ...}

    def load():
        ...                      # load the model once
        return predict           # predict(image: PIL.Image.Image) -> float

where the returned angle is the counter-clockwise rotation of the image content in
degrees (image.rotate(-angle) turns the image upright). See adapters/ for examples.
Checkpoints of this repository can be given directly with --approach and --ckpt.

The script needs numpy, opencv-python-headless and pillow, plus whatever the model
needs. It reports whether the rendered views are bit-identical to the reference
views (reference.json, with the library versions they were rendered with).

Examples:
  python eval/benchmark/run.py --adapter eval/benchmark/adapters/released_model.py \
      --out eval/benchmark/results/cgd_convnextv2_base_coco2017.json
  python eval/benchmark/run.py --approach cgd --ckpt path/to/checkpoint.ckpt \
      --name my-run --training-data "COCO 2017 train without the test photos" --out result.json
  python eval/benchmark/run.py --adapter my_adapter.py --limit 20 --out smoke.json   # quick check, not a result
"""
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import PIL
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from grid_probe import make_texture, probe_statistics, render  # noqa: E402
from image_files import list_test_images  # noqa: E402
from rotation_utils import jpeg_roundtrip, rotate_array_preserve_content  # noqa: E402

PROTOCOL = "coco-rotation-benchmark v1"
SEEDS = (0, 1, 2, 3, 4)
FORMS = {"jpeg90": 90, "lossless": None}  # JPEG quality of the save after rotation; the first form is the primary one
PROBE_CONDITIONS = ("lossless", "jpeg90", "jpeg75", "jpeg50", "lossless/jpeg90", "jpeg95/jpeg90", "jpeg75/jpeg90")
PROBE_TEXTURES = 1000
PROBE_SIZE = (640, 480)
PROBE_SEED = 0
RHO4_MAX = 0.20      # limit for the 95% upper bound of rho4
EXCESS_MAX = 0.05    # limit for the 95% upper bound of (share within 5 degrees modulo 90) - chance
INFO_KEYS = ("name", "parameters", "training_data", "input")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_photos(test_dir, limit):
    """The test photos in the pinned order, after a check of every file against photos.sha256."""
    expected = dict(reversed(line.split()) for line in (HERE / "photos.sha256").read_text().splitlines() if line.strip())
    paths = [Path(p) for p in list_test_images(test_dir)]
    names = {p.name for p in paths}
    if names != set(expected):
        raise SystemExit(f"{test_dir} must hold exactly the {len(expected)} test photos of photos.sha256 "
                         f"({len(set(expected) - names)} missing, {len(names - set(expected))} unexpected); see data/DATASETS.md")
    paths = paths[:limit] if limit else paths
    wrong = [p.name for p in paths if sha256_file(p) != expected[p.name]]
    if wrong:
        raise SystemExit(f"{len(wrong)} photos differ from the published COCO files, e.g. {wrong[:3]}")
    return paths


def load_adapter(args):
    if args.adapter:
        spec = importlib.util.spec_from_file_location("benchmark_adapter", args.adapter)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        predict = module.load()  # may complete INFO, for example with the counted parameters
        info = dict(getattr(module, "INFO", {}))
    else:
        from jpeg_gap import load_model
        import torch
        model = load_model(args.approach, args.ckpt, "cuda" if torch.cuda.is_available() else "cpu")
        info = {"name": args.name or Path(args.ckpt).stem,
                "parameters": sum(p.numel() for p in model.parameters()),
                "training_data": args.training_data,
                "input": f"{model.image_size} x {model.image_size} pixels, {model.hparams.get('model_name')}"}
        predict = model.predict_angle
    missing = [k for k in INFO_KEYS if not info.get(k)]
    if missing:
        raise SystemExit(f"the model description lacks {missing}: every result states the size of the model and its training data")
    return info, predict


class Digest:
    """One hash over all images of a set, to compare a rendering with the reference rendering."""

    def __init__(self):
        self.h = hashlib.sha256()

    def add(self, rgb):
        self.h.update(repr(rgb.shape).encode())
        self.h.update(np.ascontiguousarray(rgb).tobytes())

    def hex(self):
        return self.h.hexdigest()


def to_pil(view_bgr, digest):
    rgb = cv2.cvtColor(view_bgr, cv2.COLOR_BGR2RGB)
    digest.add(rgb)
    return Image.fromarray(rgb)


def circular_error(pred, true):
    d = np.abs(np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)) % 360.0
    return np.minimum(d, 360.0 - d)


def metrics(pred, true):
    """Statistics over all views of one set of angles."""
    e = circular_error(pred, true)
    return {"mae": float(e.mean()), "median": float(np.median(e)), "rmse": float(np.sqrt((e ** 2).mean())),
            "p90": float(np.percentile(e, 90)), "p95": float(np.percentile(e, 95)),
            "within2": float((e <= 2).mean()), "within5": float((e <= 5).mean()), "within10": float((e <= 10).mean()),
            "upside_down": int((e >= 150).sum())}


def run_test_views(predict, photos, limit):
    """Predictions for every seed and form. Returns the result block and the digests of the rendered views."""
    n_all = len((HERE / "photos.sha256").read_text().split()) // 2
    result = {form: {"per_seed": {}, "predictions": {}} for form in FORMS}
    digests, angles_out = {}, {}
    for seed in SEEDS:
        t0 = time.time()
        np.random.seed(seed)
        angles = np.random.uniform(0, 360, n_all)[:len(photos)]  # one angle per photo of the full list, in its order
        angles_out[str(seed)] = angles.tolist()
        digest = {form: Digest() for form in FORMS}
        preds = {form: [] for form in FORMS}
        for path, angle in zip(photos, angles):
            view = rotate_array_preserve_content(cv2.imread(str(path)), float(angle), mode="crop")
            for form, quality in FORMS.items():
                shown = view if quality is None else jpeg_roundtrip(view, quality)
                preds[form].append(float(predict(to_pil(shown, digest[form]))) % 360.0)
        for form in FORMS:
            result[form]["per_seed"][str(seed)] = metrics(preds[form], angles)
            result[form]["predictions"][str(seed)] = [round(p, 4) for p in preds[form]]
            digests[f"seed{seed}/{form}"] = digest[form].hex()
        m = result[next(iter(FORMS))]["per_seed"][str(seed)]
        print(f"seed {seed}: MAE {m['mae']:.2f}, within 10: {100 * m['within10']:.1f}%  ({time.time() - t0:.0f} s)", flush=True)
    for form in FORMS:
        per_seed = list(result[form]["per_seed"].values())
        result[form]["mean"] = {k: float(np.mean([s[k] for s in per_seed])) for k in per_seed[0] if k != "upside_down"}
        result[form]["mean"]["upside_down_total"] = int(sum(s["upside_down"] for s in per_seed))
    return result, digests, angles_out


def run_probe(predict, n):
    """Content-free probe, with the textures, angles and statistics of eval/grid_probe.py."""
    rng = np.random.default_rng(PROBE_SEED)
    textures = [make_texture(rng, *PROBE_SIZE) for _ in range(n)]
    angles = rng.uniform(0, 360, n)
    result, digests = {"conditions": {}, "angles": angles.tolist()}, {}
    for cond in PROBE_CONDITIONS:
        t0 = time.time()
        digest = Digest()
        preds = np.array([float(predict(to_pil(render(tex, angle, cond, "crop"), digest))) % 360.0
                          for tex, angle in zip(textures, angles)])
        stats = probe_statistics(preds, angles, seed=PROBE_SEED)
        stats["passed"] = bool(stats["rho4_ci95_upper"] <= RHO4_MAX and stats["excess_within5_ci95_upper"] <= EXCESS_MAX)
        stats["predictions"] = [round(float(p), 4) for p in preds]
        result["conditions"][cond] = stats
        digests[f"probe/{cond}"] = digest.hex()
        print(f"probe {cond:>15}: within 5 deg mod 90: {100 * stats['mod90_within5']:5.1f}% (chance 11.1%), "
              f"rho4 {stats['rho4']:+.3f} (upper bound {stats['rho4_ci95_upper']:+.3f}), "
              f"{'passed' if stats['passed'] else 'NOT PASSED'}  ({time.time() - t0:.0f} s)", flush=True)
    result["passed"] = sum(c["passed"] for c in result["conditions"].values())
    result["all_passed"] = result["passed"] == len(PROBE_CONDITIONS)
    return result, digests


def environment():
    from PIL import features
    return {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
            "pillow": PIL.__version__, "libjpeg_turbo": features.version("libjpeg_turbo") or features.version("jpg")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", help="Python file that connects a model (INFO and load(), see above)")
    ap.add_argument("--approach", help="for a checkpoint of this repository: cgd, classification, unit_vector, psc or direct_angle")
    ap.add_argument("--ckpt", help="for a checkpoint of this repository: the checkpoint file")
    ap.add_argument("--name", help="with --ckpt: name of the model in the result (default: the file name)")
    ap.add_argument("--training-data", help="with --ckpt: what the model was trained on")
    ap.add_argument("--test-dir", default=str(ROOT / "data" / "datasets" / "test_coco"), help="the 1,030 test photos")
    ap.add_argument("--limit", type=int, default=0, help="first N photos and N textures only: a quick check, not a result")
    ap.add_argument("--write-reference", action="store_true", help="store the digests of this rendering as the reference")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if bool(args.adapter) == bool(args.approach and args.ckpt):
        ap.error("give either --adapter, or --approach together with --ckpt")

    photos = check_photos(args.test_dir, args.limit)
    info, predict = load_adapter(args)
    print(f"{PROTOCOL}: {info['name']} ({info['parameters']:,} parameters)", flush=True)

    test, digests, angles = run_test_views(predict, photos, args.limit)
    probe, probe_digests = run_probe(predict, args.limit or PROBE_TEXTURES)
    digests.update(probe_digests)

    reference_file = HERE / "reference.json"
    identical = None
    if args.limit:
        print("\n--limit was given: this is a quick check and not a benchmark result")
    elif args.write_reference:
        reference_file.write_text(json.dumps({"protocol": PROTOCOL, "environment": environment(), "digests": digests}, indent=1) + "\n")
        print(f"\nwrote {reference_file}")
        identical = True
    elif reference_file.exists():
        reference = json.loads(reference_file.read_text())
        differing = [k for k, v in reference["digests"].items() if digests.get(k) != v]
        identical = not differing
        if differing:
            print(f"\nNote: {len(differing)} of {len(digests)} image sets are not bit-identical to the reference rendering "
                  f"(reference: {reference['environment']}; here: {environment()}).\n"
                  "The images are built the same way, but other library versions round some pixels differently.")
        else:
            print("\nAll images are bit-identical to the reference rendering.")

    result = {"protocol": PROTOCOL, "complete": not args.limit, "model": info, "environment": environment(),
              "images_identical_to_reference": identical, "test": test, "angles": angles, "probe": probe, "digests": digests}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")

    print(f"\n{'form':>9} {'MAE':>6} {'median':>7} {'P90':>6} {'P95':>6} {'within 2':>9} {'within 5':>9} {'within 10':>10} {'upside down':>12}")
    for form, block in test.items():
        m = block["mean"]
        print(f"{form:>9} {m['mae']:6.2f} {m['median']:7.2f} {m['p90']:6.2f} {m['p95']:6.2f} {100 * m['within2']:8.1f}% "
              f"{100 * m['within5']:8.1f}% {100 * m['within10']:9.1f}% {m['upside_down_total']:>12}")
    print(f"processing traces: {probe['passed']} of {len(PROBE_CONDITIONS)} probe conditions passed")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
