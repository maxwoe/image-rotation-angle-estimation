"""Test that the code example from the HF model card works end-to-end.

Run from a temp directory to avoid snapshot_download overwriting local files:
    cp hf_space/*.py hf_space/config.json /tmp/test_hf/ && cp -r hf_space/examples /tmp/test_hf/
    cd /tmp/test_hf && python test_model_card.py
"""
from model_cgd import CGDAngleEstimation
from PIL import Image

# Load model (defaults to COCO 2017 checkpoint)
model = CGDAngleEstimation.from_pretrained("maxwoe/image-rotation-angle-estimation")

# Predict rotation angle
image = Image.open("examples/COCO_val2014_000000168337.jpg")
angle = model.predict_angle(image)
print(f"Predicted rotation: {angle:.2f} degrees")
