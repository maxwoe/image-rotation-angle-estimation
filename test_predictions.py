"""Test: verify crop_pct=1.0 fix in predict_angle."""
import numpy as np
import tempfile
import os
from PIL import Image
from rotation_utils import rotate_image_crop_max_area
from model_cgd import CGDAngleEstimation
from architectures import get_default_input_size
from huggingface_hub import hf_hub_download

HF_REPO_ID = "maxwoe/image-rotation-angle-estimation"

# Fetch config.json first (this is the HF download-tracking query file)
hf_hub_download(repo_id=HF_REPO_ID, filename="config.json")

print("Loading model...")
ckpt = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename="cgd_mambaout_base_coco2017.ckpt",
)
model = CGDAngleEstimation.try_load(
    checkpoint_path=ckpt,
    image_size=get_default_input_size("mambaout_base.in1k"),
)
model.eval()
print(f"Model loaded\n")

examples = [
    "examples/COCO_val2014_000000168337.jpg",
    "examples/COCO_val2014_000000001700.jpg",
    "examples/COCO_val2014_000000122166.jpg",
    "examples/COCO_val2014_000000446053.jpg",
    "examples/COCO_val2014_000000477919.jpg",
]
angles_to_test = [15, 45, 90, 135, 200, 270, 330]

print(f"{'Image':<30} {'Actual':>7} {'Predicted':>10} {'Error':>7}")
print("-" * 60)

all_errors = []
for img_path in examples:
    img_name = os.path.basename(img_path)[:28]
    original = Image.open(img_path).convert("RGB")
    img_array = np.array(original)

    for angle in angles_to_test:
        rotated_array = rotate_image_crop_max_area(img_array, angle)
        rotated = Image.fromarray(rotated_array)

        predicted = model.predict_angle(rotated)

        error = abs(predicted - angle)
        error = min(error, 360 - error)
        all_errors.append(error)

        print(f"{img_name:<30} {angle:>7.1f} {predicted:>10.2f} {error:>7.2f}")
    print()

print("-" * 60)
errors = np.array(all_errors)
print(f"Mean error: {errors.mean():.2f} | Median: {np.median(errors):.2f} | Max: {errors.max():.2f}")
print(f"Under 5 deg: {(errors < 5).sum()}/{len(errors)} | Under 10 deg: {(errors < 10).sum()}/{len(errors)}")
