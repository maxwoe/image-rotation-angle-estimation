---
license: mit
tags:
  - image-rotation
  - orientation-estimation
  - angle-detection
  - circular-gaussian-distribution
  - mambaout
  - pytorch
datasets:
  - coco
metrics:
  - mae
pipeline_tag: image-classification
---

# Image Rotation Angle Estimation

**[Try the interactive demo](https://huggingface.co/spaces/maxwoe/image-rotation-angle-estimation)** | **[GitHub](https://github.com/maxwoe/image-rotation-angle-estimation)** | **[Paper](https://arxiv.org/abs/2603.25351)**

Predicts the rotation angle of an image using the **Circular Gaussian Distribution (CGD)** method with a **MambaOut Base** backbone.

The model outputs a probability distribution over 360 angle bins (1 degree resolution) and extracts the predicted angle via argmax. It handles the full 360 degree range with no boundary discontinuities.

## Available Checkpoints

| Checkpoint | Dataset | MAE | Median Error |
|---|---|---|---|
| `cgd_mambaout_base_coco2017.ckpt` | COCO 2017 | 2.84° | 0.55° |
| `cgd_mambaout_base_coco2014.ckpt` | COCO 2014 | 3.71° | 0.68° |

## Usage

Download the inference code from this Hub repo (`model_cgd.py`, `architectures.py`, `rotation_utils.py`), then:

```python
from model_cgd import CGDAngleEstimation
from PIL import Image

# Load model (defaults to COCO 2017 checkpoint)
model = CGDAngleEstimation.from_pretrained("maxwoe/image-rotation-angle-estimation")

# Or load a specific checkpoint
# model = CGDAngleEstimation.from_pretrained(
#     "maxwoe/image-rotation-angle-estimation",
#     model_name="cgd_mambaout_base_coco2014.ckpt",
# )

image = Image.open("your_image.jpg")
angle = model.predict_angle(image)
print(f"Predicted rotation: {angle:.1f}°")
```

`predict_angle` accepts a PIL Image, numpy array, or file path.

## Evaluation Results (COCO 2017, 5 seeds)

| Metric | Value |
|---|---|
| MAE | 2.84° |
| Median Error | 0.55° |
| RMSE | 8.45° |
| P90 Error | 3.54° |
| P95 Error | 12.00° |
| Accuracy at 2° | 90.2% |
| Accuracy at 5° | 97.5% |
| Accuracy at 10° | 98.1% |

## Model Details

- **Method:** Circular Gaussian Distribution (CGD), 360 bins, sigma = 6.0°
- **Backbone:** MambaOut Base (`mambaout_base.in1k`), pretrained on ImageNet-1K
- **Input size:** 224 x 224 pixels
- **Output:** Probability distribution over 360 angle bins, converted to angle via argmax
- **Loss:** KL Divergence with soft Gaussian labels
- **Optimizer:** AdamW with ReduceLROnPlateau scheduler

## License

MIT
