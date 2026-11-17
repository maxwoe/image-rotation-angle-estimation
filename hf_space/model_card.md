---
license: mit
base_model: timm/convnextv2_base.fcmae_ft_in22k_in1k
tags:
  - image-rotation
  - rotation-estimation
  - orientation-estimation
  - angle-detection
  - circular-gaussian-distribution
  - convnextv2
  - pytorch
datasets:
  - coco
metrics:
  - mae
pipeline_tag: image-classification
model-index:
  - name: cgd_convnextv2_base_coco2017
    results:
      - task:
          type: image-classification
          name: Image rotation angle estimation
        dataset:
          type: coco
          name: COCO test photos (1,030 images), rotated and saved as JPEG
        metrics:
          - type: mae
            value: 3.71
            name: Mean absolute error (degrees)
---

# Image Rotation Angle Estimation

**[Try the interactive demo](https://huggingface.co/spaces/maxwoe/image-rotation-angle-estimation)** | **[GitHub](https://github.com/maxwoe/image-rotation-angle-estimation)** | **[Paper](https://arxiv.org/abs/2603.25351)**

Estimates by how many degrees a photo is rotated away from upright, anywhere in the full 360° range, so that it can be turned back.
On COCO test photos the mean absolute error is 3.71° and 97% of the predictions are within 10°.
With the simpler decoding of the paper, the same checkpoint reaches 3.77° (see [Evaluation](#evaluation)).

![Five rotated photos and the same photos turned back by the predicted angle](example.jpg)

The model is a **ConvNeXt V2 Base** backbone with a **Circular Gaussian Distribution (CGD)** head.
It outputs a probability distribution over 360 one-degree bins and reads the angle from the bins around the most probable one, so there is no discontinuity at 0°/360°.

## Usage

```bash
pip install torch torchvision pytorch-lightning timm huggingface_hub pillow numpy loguru
```

```python
import sys
from huggingface_hub import snapshot_download
from PIL import Image

# The inference code is part of this repository
sys.path.insert(0, snapshot_download("maxwoe/image-rotation-angle-estimation", allow_patterns=["*.py"]))
from model_cgd import CGDAngleEstimation

model = CGDAngleEstimation.from_pretrained("maxwoe/image-rotation-angle-estimation")

image = Image.open("your_image.jpg")
angle = model.predict_angle(image)  # degrees in [0, 360), counter-clockwise
print(f"Predicted rotation: {angle:.1f}°")

upright = image.rotate(-angle, expand=True)  # turn the image back

# optionally with a confidence between 0 and 1, see "Confidence" below
angle, confidence = model.predict_angle(image, return_confidence=True)
```

- `predict_angle` accepts a PIL Image, a numpy array or a file path.
- The angle is measured counter-clockwise: 90° means the image was turned a quarter turn to the left, and rotating it by `-angle` restores it.
- Inference takes about 0.3 seconds per image on a CPU; call `model.to("cuda")` to use a GPU.

## Evaluation

1,030 COCO test photos, each rotated by a random angle and cropped to the largest rectangle that contains no border; means over five sets of random angles.
In the first column the rotated image is saved once as JPEG (quality 90) before the model sees it, which is how rotated photos exist in practice.
In the second column it is passed to the model without ever being saved.
The test follows the [benchmark protocol](https://github.com/maxwoe/image-rotation-angle-estimation/blob/main/BENCHMARK.md) of the repository, which any model can be scored on.

| Metric | Saved as JPEG after rotation | Never saved after rotation |
|---|---|---|
| Mean absolute error | 3.71° | 4.38° |
| Median error | 1.11° | 1.33° |
| Within 2° | 73.8% | 66.4% |
| Within 5° | 93.9% | 90.1% |
| Within 10° | 97.3% | 95.9% |
| 90th percentile error | 3.74° | 4.99° |
| 95th percentile error | 5.60° | 8.24° |
| RMSE | 17.32° | 18.24° |

The error is nearly the same across JPEG quality settings (3.81° at quality 95, 3.69° at quality 75).

The paper uses a simpler decoding, so its numbers are slightly higher than the ones above.
It takes the most probable bin as the angle (argmax), which limits the answer to whole degrees.
The released code takes the circular mean of the bins within 10° of that bin.
The network and its weights are the same in both cases:

| Decoding | Saved as JPEG after rotation | Never saved after rotation |
|---|---|---|
| Most probable bin, as in the paper | 3.77° | 4.45° |
| Mean of the bins within 10°, as in the released code and on this card | 3.71° | 4.38° |

The paper gives the configuration as the mean of three training runs (3.76° and 4.45°), of which this checkpoint is one.
Its appendix on decoding reports the same effect for a soft decoding over all bins (3.76° against 3.68°).

## Comparison with other models

Two other open models estimate the rotation angle over the full circle.
[OAD-360](https://github.com/pidahbus/deep-image-orientation-angle-detection) (Maji and Bose, 2020) is the strongest earlier model in the paper, and it was scored for the paper on the test views of this card.
[RightWayUp](https://github.com/ortusaitech/rightwayup) (ORTUS AI, September 2026) is a family of models whose output head follows the CGD formulation used here, and ORTUS AI published results of its models on the same test photos.
The table shows means or totals over five sets of random angles (5,150 views).

| | This model | RightWayUp Max | RightWayUp Fast | OAD-360 |
|---|---|---|---|---|
| Parameters | 88M | 306M | 23M | 86M |
| Training images | 117,372 | about 1.24 million | about 1.24 million | most of COCO |
| Median error, saved as JPEG | 1.11° | 1.21° | not published | 2.08° |
| Within 2°, saved as JPEG | 73.8% | 69.2% | not published | 48.7% |
| Within 10°, saved as JPEG after rotation | 97.3% | 98.4% | 96.4% | 93.0% |
| Within 10°, never saved after rotation | 95.9% | 98.8% | 96.5% | 93.1% |
| More than 10° off, saved as JPEG | 140 | 83 | not published | 361 |
| Upside down (150° or more off), saved as JPEG | 35 | 5 | not published | not recorded |
| Mean error, saved as JPEG | 3.71° | 2.13° | not published | 6.09° |
| Trace check, conditions passed | 7 of 7 | not run | not run | not run |

On a typical photo saved as JPEG this model is the most precise of the three models with published numbers: it has the lowest median error and the largest share of predictions within 2°.
It reaches this with less than a third of the parameters of RightWayUp Max and a tenth of its training images.
RightWayUp Max makes the fewest large errors, which is why its mean error is the lowest.
The trace check is the one of the [benchmark protocol](https://github.com/maxwoe/image-rotation-angle-estimation/blob/main/BENCHMARK.md).
The RightWayUp models pass a check of ORTUS AI's own on images without content.

The RightWayUp numbers are taken from its [benchmark notes](https://github.com/ortusaitech/rightwayup/blob/main/benchmarks/woehrer-2026/BENCHMARK-NOTES.md).
ORTUS AI builds the views with the same code but pairs the photos with the angles in another order, so the totals over the five sets are comparable and single views are not.
The paper itself compares five methods on three backbones under the same protocol.

## Confidence

`predict_angle(image, return_confidence=True)` returns the angle together with a confidence between 0 and 1.
The confidence estimates the probability that the predicted angle is within 10° of the true angle.
It is computed from the output of the model: the probability mass within 10° of the predicted angle, mapped to the range from 0 to 1 by a logistic curve with two parameters that were fitted on the validation split.
A second peak in the distribution, for example at the upside-down angle, lowers it.

The model is right on most photos, so most values are close to 1: 84% of the test views get 0.99 or more.
The confidence is therefore best used with a threshold, below which a prediction is not trusted.
The table shows the effect on the test views that were saved as JPEG after rotation.

| Confidence at least | Views answered | Within 10° among those | Mean absolute error among those |
|---|---|---|---|
| 0.50 | 99.0% | 97.8% | 3.25° |
| 0.90 | 97.3% | 98.4% | 2.85° |
| 0.95 | 95.9% | 98.7% | 2.57° |
| 0.98 | 92.4% | 99.3% | 2.07° |
| 0.99 | 84.4% | 99.6% | 1.74° |

On views that were never saved after rotation, the same thresholds answer fewer views: 0.98 answers 86.5% of them, with 98.9% within 10°.
On noise images that have no orientation at all, the median confidence is 0.09 and at most 3% of them reach 0.9.

What the confidence cannot do:

- **It is not an exact percentage.** It is reliable at the top and too optimistic in the middle: of the predictions with a confidence between 0.95 and 0.98, 84% are within 10°.
- **Confident mistakes remain.** Of the 87 test views with an error above 45°, which are mostly confusions by 90° or 180°, 49% have a confidence of at least 0.95 and 30% of at least 0.98.
- **Fitted on photos.** The curve was fitted and tested on COCO photos, and the values were not evaluated on other kinds of images.
- **Default checkpoint only.** The legacy checkpoints have no confidence value.

## Limitations

- **Photos only.** The model was trained and tested on everyday photos (COCO). Documents, scans, screenshots, drawings, aerial and medical images were not evaluated.
- **Ambiguous images.** Close-ups, textures and symmetric scenes have no clear "up". More than half of the large errors are confusions by 90° or 180° on such images.
- **Central square.** The image is resized so that its shorter side is 224 pixels and the central 224 x 224 square is used. In a very elongated image the model does not see the ends.
- **Images never saved after rotation.** Rotations that are applied in memory or kept in a lossless format are slightly outside the training distribution, which is why the error is a little higher in the second column above.
- **Confidence is not a guarantee.** The confidence value flags most wrong predictions, but not all of them (see Confidence).

## Checks for Processing Traces

A model for this task can score well without looking at the image content: when a decoded JPEG photo is rotated, the 8x8 block grid of the JPEG codec turns with it and reveals the angle.
This model was trained under the trace-free protocol of the paper, in which every training image is saved as JPEG after rotation, and it was then tested for reliance on such traces:

- **Content-free probe.** Noise images without any orientation, with and without a JPEG grid, are rotated and shown to the model, which should then be at chance. It meets both criteria of the probe in all seven conditions. In the closest one, 13.5% of its predictions land within 5° of the grid angle (chance: 11.1%), and the upper confidence bound of this excess is 4.5 points, against a limit of 5 points fixed in advance. With argmax decoding the bound in that condition is 5.1 points, so the model then misses it by 0.1 point.
- **Cue conflict.** When a grid is stamped at an angle that contradicts the content, the model follows the content; the grid attracts at most 6% of the predictions.
- **Test photos.** A rotated source grid changes the share of predictions within 2° by -0.4 points, and the interpolation pattern of the rotation by 1.5 points; both are inside the limit of 5 points.

The cue conflict and the test on photos were measured with argmax decoding.
The scripts for these checks are in the GitHub repository.

## Checkpoints

| Checkpoint | Training data | Error, saved as JPEG after rotation | Error, never saved after rotation |
|---|---|---|---|
| `cgd_convnextv2_base_coco2017.ckpt` (default) | COCO 2017 | 3.71° | 4.38° |
| `cgd_mambaout_base_coco2017.ckpt` (legacy) | COCO 2017 | 24.45° | 2.84° |
| `cgd_mambaout_base_coco2014.ckpt` (legacy) | COCO 2014 | 21.05° | 3.71° |

The legacy checkpoints are decoded with the most probable bin, as in the paper.

**Do not use the legacy checkpoints for real photos.**
They were trained on images that were rotated but never saved again, and they learned to read the rotated JPEG grid described above.
That makes them very precise on images that were never saved after rotation and unreliable on everything else.
They remain available so that earlier results stay reproducible, and can be loaded with `from_pretrained(..., model_name="cgd_mambaout_base_coco2017.ckpt")`.

## Model Details

- **Method:** Circular Gaussian Distribution (CGD), 360 bins, sigma = 6.0°, KL divergence against soft Gaussian labels
- **Decoding:** circular mean of the bins within 10° of the most probable bin (the legacy checkpoints and the paper use argmax)
- **Backbone:** ConvNeXt V2 Base (`convnextv2_base.fcmae_ft_in22k_in1k`), pretrained on ImageNet, fine-tuned end to end
- **Size:** 88 million parameters, 352 MB (weights only)
- **Input:** 224 x 224 pixels, ImageNet normalization
- **Training data:** COCO 2017 train split without the test images (117,372 images), 10% held out for validation
- **Training images:** rotated by a random angle in [0°, 360°), cropped to the largest rectangle without border, saved as JPEG at a random quality between 40 and 90; the interpolation method of the rotation is drawn at random
- **Optimizer:** AdamW, learning rate reduced on plateau, early stopping
- **Confidence:** probability mass within 10° of the predicted angle, mapped by a logistic curve on its logit (slope 2.998, intercept -1.344) that was fitted on the validation split

## Paper and Citation

The method comparison behind this model is described in [Image Rotation Angle Estimation: Five Circular-Aware Methods Across Sixteen Architectures](https://arxiv.org/abs/2603.25351).
The first arXiv version (v1) predates the trace-free protocol: its COCO results (for example 2.84°) are those of the legacy checkpoints.
Version 2 (17 November 2026) describes the protocol and the model published here.
Its numbers use argmax decoding and are therefore slightly higher than the ones on this card (see [Evaluation](#evaluation)).

```bibtex
@misc{woehrer2026irae,
  title={Image Rotation Angle Estimation: Five Circular-Aware Methods Across Sixteen Architectures},
  author={Woehrer, Maximilian},
  year={2026},
  eprint={2603.25351},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.25351}
}
```

## License

MIT
