"""Predictions of the default Hub model on rotated example images.

Downloads the default checkpoint named in the Hub's config.json and predicts the angle of
the example images at a set of known rotations, on views that were never re-encoded and
on the same views saved once as JPEG at quality 90. A model that reads the image content
gives about the same errors in both forms.

Usage (from the repository root):
    python test_predictions.py
"""
import glob
import io
import json
import os

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from model_cgd import CGDAngleEstimation
from rotation_utils import rotate_image_crop_max_area

HF_REPO_ID = "maxwoe/image-rotation-angle-estimation"

# config.json names the default model (it is also the HF download-tracking query file)
with open(hf_hub_download(repo_id=HF_REPO_ID, filename="config.json"), encoding="utf-8") as f:
    config = json.load(f)
info = config["models"][config["default_model"]]

print(f"Loading {info['filename']}...")
ckpt = hf_hub_download(repo_id=HF_REPO_ID, filename=info["filename"])
model = CGDAngleEstimation.try_load(checkpoint_path=ckpt, image_size=info["input_size"])
model.eval()
print("Model loaded\n")

examples = sorted(glob.glob("hf_space/examples/*.jpg"))
angles_to_test = [15, 45, 90, 135, 200, 270, 330]


def circular_error(predicted, actual):
    error = abs(predicted - actual)
    return min(error, 360 - error)


print(f"{'Image':<30} {'Actual':>7} {'Predicted':>10} {'Error':>7} {'JPEG q90':>9} {'Error':>7}")
print("-" * 76)

errors, errors_jpeg = [], []
for img_path in examples:
    img_name = os.path.basename(img_path)[:28]
    img_array = np.array(Image.open(img_path).convert("RGB"))

    for angle in angles_to_test:
        rotated = Image.fromarray(rotate_image_crop_max_area(img_array, angle))
        predicted = model.predict_angle(rotated)

        buffer = io.BytesIO()
        rotated.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        predicted_jpeg = model.predict_angle(Image.open(buffer))

        errors.append(circular_error(predicted, angle))
        errors_jpeg.append(circular_error(predicted_jpeg, angle))
        print(f"{img_name:<30} {angle:>7.1f} {predicted:>10.2f} {errors[-1]:>7.2f} "
              f"{predicted_jpeg:>9.2f} {errors_jpeg[-1]:>7.2f}")
    print()

print("-" * 76)
for label, values in (("never re-encoded", np.array(errors)), ("JPEG q90", np.array(errors_jpeg))):
    print(f"{label:<17} mean error {values.mean():.2f} | median {np.median(values):.2f} | max {values.max():.2f} | "
          f"under 5 deg {(values < 5).sum()}/{len(values)} | under 10 deg {(values < 10).sum()}/{len(values)}")
