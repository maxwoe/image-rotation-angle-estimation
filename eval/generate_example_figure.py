#!/usr/bin/env python3
"""
Before/after figure for the model card and the README.

Each example photo of the demo is rotated by a fixed angle, cropped to the largest
rectangle without border and saved once as JPEG, as a rotated photo exists in practice.
The model predicts the angle, and the image is turned back by the predicted angle, as
the usage example of the model card does. The angles are fixed in advance and every
example is shown, whatever the error.

    python eval/generate_example_figure.py [--ckpt weights/cgd_convnextv2_base_coco2017.ckpt]
        [--output hf_space/example.jpg]
"""
import argparse
import glob
import io
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model_cgd import CGDAngleEstimation  # noqa: E402
from rotation_utils import rotate_image_crop_max_area  # noqa: E402

ANGLES = [40, 115, 180, 250, 325]  # one per example photo, in file order


CELL = 640  # every image is fitted into a white square of this size, so the columns align


def turn_back(view, angle):
    """Rotate a view by -angle, keeping all of it, as image.rotate(-angle, expand=True) does."""
    return view.rotate(-angle, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def fit_square(image):
    scale = CELL / max(image.size)
    image = image.resize((round(image.size[0] * scale), round(image.size[1] * scale)), Image.LANCZOS)
    canvas = Image.new("RGB", (CELL, CELL), (255, 255, 255))
    canvas.paste(image, ((CELL - image.size[0]) // 2, (CELL - image.size[1]) // 2))
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="weights/cgd_convnextv2_base_coco2017.ckpt")
    ap.add_argument("--examples", default="hf_space/examples")
    ap.add_argument("--output", default="hf_space/example.jpg")
    args = ap.parse_args()

    model = CGDAngleEstimation.try_load(checkpoint_path=args.ckpt, image_size=224)
    model.eval()

    columns = []
    for path, angle in zip(sorted(glob.glob(os.path.join(args.examples, "*.jpg"))), ANGLES):
        photo = np.array(Image.open(path).convert("RGB"))
        buffer = io.BytesIO()
        Image.fromarray(rotate_image_crop_max_area(photo, angle)).save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        view = Image.open(buffer).convert("RGB")
        predicted = model.predict_angle(view)
        error = min(abs(predicted - angle), 360 - abs(predicted - angle))
        print(f"{os.path.basename(path)}: rotated by {angle}, predicted {predicted:.0f}, error {error:.0f}")
        columns.append((view, turn_back(view, predicted), angle, predicted))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#1f2328", "#57606a"
    n = len(columns)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.9), dpi=200)
    fig.patch.set_facecolor("white")
    for col, (view, corrected, angle, predicted) in enumerate(columns):
        for row, image in enumerate((view, corrected)):
            ax = axes[row, col]
            ax.imshow(fit_square(image))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[0, col].set_title(f"rotated by {angle}°", fontsize=9, color=muted, pad=5)
        axes[1, col].set_title(f"predicted {predicted:.0f}°", fontsize=9, color=muted, pad=5)
    fig.text(0.014, 0.72, "Input", rotation=90, va="center", ha="center", fontsize=11, color=ink, fontweight="bold")
    fig.text(0.014, 0.25, "Turned back", rotation=90, va="center", ha="center", fontsize=11, color=ink, fontweight="bold")
    fig.subplots_adjust(left=0.04, right=0.995, top=0.945, bottom=0.01, wspace=0.06, hspace=0.12)

    canvas = io.BytesIO()
    fig.savefig(canvas, format="png", facecolor="white")
    canvas.seek(0)
    # high quality and no chroma subsampling keep the captions sharp
    Image.open(canvas).convert("RGB").save(args.output, quality=93, subsampling=0)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
