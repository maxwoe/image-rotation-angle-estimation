---
license: mit
tags:
  - image-rotation
  - orientation-estimation
  - angle-detection
  - circular-gaussian-distribution
  - mambaout
datasets:
  - coco
metrics:
  - mae
pipeline_tag: image-classification
library_name: pytorch
---

# Image Rotation Angle Estimation

**[Try the interactive demo](https://huggingface.co/spaces/maxwoe/image-rotation-angle-estimation)**

Predicts the rotation angle of an image using the **Circular Gaussian Distribution (CGD)** method with a **MambaOut Base** backbone.

The model outputs a probability distribution over 360 angle bins (1 degree resolution) and extracts the predicted angle via argmax. It handles the full 360 degree range with no boundary discontinuities.

## Available Checkpoints

| Checkpoint | Dataset | MAE | Median Error |
|---|---|---|---|
| `cgd_mambaout_base_coco2017.ckpt` | COCO 2017 | 2.84 deg | 0.55 deg |
| `cgd_mambaout_base_coco2014.ckpt` | COCO 2014 | 3.71 deg | 0.68 deg |

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

# Download checkpoint
ckpt_path = hf_hub_download(
    repo_id="maxwoe/image-rotation-angle-estimation",
    filename="cgd_mambaout_base_coco2017.ckpt",
)

# Load model
from model_cgd import CGDAngleEstimation

model = CGDAngleEstimation.load_from_checkpoint(ckpt_path, image_size=224)
model.eval()

# Predict rotation angle
image = Image.open("your_image.jpg")
angle = model.predict_angle(image)
print(f"Predicted rotation: {angle:.2f} degrees")
```

## Evaluation Results (COCO 2017, 5 seeds)

| Metric | Value |
|---|---|
| MAE | 2.84 deg |
| Median Error | 0.55 deg |
| RMSE | 8.45 deg |
| P90 Error | 3.54 deg |
| P95 Error | 12.00 deg |
| Accuracy at 2 deg | 90.2% |
| Accuracy at 5 deg | 97.5% |
| Accuracy at 10 deg | 98.1% |

## Model Details

- **Method:** Circular Gaussian Distribution (CGD) — 360 bins, sigma = 6.0 degrees
- **Backbone:** MambaOut Base (`mambaout_base.in1k`), pretrained on ImageNet-1K
- **Input size:** 224 x 224 pixels
- **Output:** Probability distribution over 360 angle bins, converted to angle via argmax
- **Loss:** KL Divergence with soft Gaussian labels
- **Optimizer:** AdamW with ReduceLROnPlateau scheduler

## License

MIT
