"""HuggingFace Spaces demo for Image Rotation Angle Estimation.

Two-step interactive demo:
1. Upload an image and click "Random Rotate" to apply a random rotation
2. Click "Correct Orientation" to see the model predict and correct the angle
"""

import gradio as gr
import torch
from PIL import Image
import os
import random
import tempfile
import numpy as np
from loguru import logger
from huggingface_hub import hf_hub_download

from model_cgd import CGDAngleEstimation
from architectures import get_default_input_size
from rotation_utils import rotate_image_crop_max_area

# HuggingFace Hub configuration
HF_REPO_ID = os.environ.get("HF_MODEL_REPO", "maxwoe/image-rotation-angle-estimation")
HF_MODELS = {
    "CGD + MambaOut Base (COCO 2017) — 2.84° MAE": {
        "filename": "cgd_mambaout_base_coco2017.ckpt",
        "architecture": "mambaout_base.in1k",
    },
    "CGD + MambaOut Base (COCO 2014) — 3.71° MAE": {
        "filename": "cgd_mambaout_base_coco2014.ckpt",
        "architecture": "mambaout_base.in1k",
    },
}
HF_DEFAULT_MODEL = "CGD + MambaOut Base (COCO 2017) — 2.84° MAE"

# Global model state
model = None
current_model_name = None


def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model(name):
    """Download and load a model from HuggingFace Hub."""
    global model, current_model_name
    if name == current_model_name and model is not None:
        return gr.Info(f"Model already loaded: {name}")

    if name not in HF_MODELS:
        return gr.Warning(f"Unknown model: {name}")

    info = HF_MODELS[name]
    logger.info(f"Downloading {info['filename']} from {HF_REPO_ID}...")
    local_path = hf_hub_download(repo_id=HF_REPO_ID, filename=info["filename"])

    architecture = info["architecture"]
    image_size = get_default_input_size(architecture)

    logger.info(f"Loading model from {local_path}...")
    new_model = CGDAngleEstimation.try_load(checkpoint_path=local_path, image_size=image_size)
    new_model.eval()

    device = get_device()
    if device.startswith("cuda"):
        new_model = new_model.to(device)

    model = new_model
    current_model_name = name
    logger.info(f"Model loaded: {name} on {device}")
    return gr.Info(f"Loaded: {name} ({device})")


def store_original(image):
    """Store the uploaded image as the original for rotation."""
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return image


def random_rotate(original):
    """Apply a random rotation to the original uploaded image."""
    if original is None:
        return None, None

    angle = random.uniform(0, 360)
    img_array = np.array(original)
    rotated_array = rotate_image_crop_max_area(img_array, angle)
    rotated = Image.fromarray(rotated_array)
    return rotated, angle


def correct_orientation(image):
    """Predict the rotation angle and correct the image."""
    if image is None:
        return None, "Please upload and rotate an image first."
    if model is None:
        return None, "Model is still loading, please wait..."

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Save to temp file (predict_angle expects a file path)
    temp_fd, temp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(temp_fd)
    image.convert("RGB").save(temp_path)

    try:
        predicted_angle = model.predict_angle(temp_path)
    finally:
        os.unlink(temp_path)

    corrected = image.rotate(-predicted_angle, expand=True, fillcolor=(255, 255, 255))

    return corrected, f"Predicted rotation: {predicted_angle:.2f}°"


# Build UI
app = gr.Blocks(title="Image Rotation Angle Estimation")
with app:
    gr.HTML("<h1>Image Rotation Angle Estimation</h1>")
    gr.Markdown(
        "Upload an image, apply a random rotation, and see the model predict and correct the angle.\n\n"
        "Uses the **CGD** (Circular Gaussian Distribution) method with **MambaOut Base** architecture. "
        # "See the [paper and code](https://github.com/maxwoe/image-rotation-angle-estimation) for details."
    )

    original_image_state = gr.State(value=None)
    actual_angle_state = gr.State(value=None)

    model_dropdown = gr.Dropdown(
        choices=list(HF_MODELS.keys()),
        value=HF_DEFAULT_MODEL,
        label="Model",
    )
    model_dropdown.change(load_model, inputs=[model_dropdown])

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image", height=400)
            rotate_btn = gr.Button("Random Rotate", variant="secondary", size="lg")
        with gr.Column():
            corrected_image = gr.Image(label="Corrected Image", height=400, interactive=False)
            result_text = gr.Textbox(label="Prediction Result", lines=1, interactive=False)

    correct_btn = gr.Button("Correct Orientation", variant="primary", size="lg")

    input_image.upload(
        store_original,
        inputs=[input_image],
        outputs=[original_image_state],
    )
    rotate_btn.click(
        random_rotate,
        inputs=[original_image_state],
        outputs=[input_image, actual_angle_state],
    )
    correct_btn.click(
        correct_orientation,
        inputs=[input_image],
        outputs=[corrected_image, result_text],
    )

    app.load(lambda: load_model(HF_DEFAULT_MODEL))

app.launch()
