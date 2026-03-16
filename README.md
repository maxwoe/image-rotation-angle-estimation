# Image Rotation Angle Estimation

Systematic comparison of five circular-aware methods for image rotation angle estimation across sixteen architectures. Accompanies the paper:

> **Image Rotation Angle Estimation: Comparing Circular-Aware Methods**

## Methods

| Abbreviation | Approach | Description |
|---|---|---|
| DA | Direct Angle | Scalar regression with circular loss |
| CLS | Classification | Angular binning (360 bins) |
| UV | Unit Vector | Regression on (cos θ, sin θ) |
| PSC | Phase-Shifting Coder | Frequency-domain encoding |
| CGD | Circular Gaussian Distribution | Probabilistic soft targets via KL divergence |

## Results

Best results on the DRC-D test set (mean of 5 seeds):

| Method | Architecture | MAE (°) |
|---|---|---|
| CLS | EfficientViT-B3 | 1.23 |
| CGD | MambaOut Base | 1.24 |

Transfer to COCO (CGD + MambaOut Base):

| Training Data | MAE (°) |
|---|---|
| COCO 2014 | 3.71 |
| COCO 2017 | 2.84 |

## Pretrained Models

Pretrained checkpoints (CGD + MambaOut Base) are available on the [HuggingFace Hub](https://huggingface.co/maxwoe/image-rotation-angle-estimation):

| Checkpoint | Training Data | MAE (°) |
|---|---|---|
| `cgd_mambaout_base_coco2017.ckpt` | COCO 2017 | 2.84 |
| `cgd_mambaout_base_coco2014.ckpt` | COCO 2014 | 3.71 |

Try the interactive demo on [HuggingFace Spaces](https://huggingface.co/spaces/maxwoe/image-rotation-angle-estimation).

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+ and CUDA 11.8+.

## Datasets

### DRC-D

Download from [Google Drive](https://drive.google.com/drive/folders/1y8964QKakL1zJsuzuivCx41_YkrsOKv_) (original source: [RotationCorrection](https://github.com/nie-lang/RotationCorrection)).

Extract into `data/datasets/ds_drcd/`:
```
data/datasets/ds_drcd/
├── training/
│   ├── input/    # rotated images (multiple rotations per source image)
│   └── gt/       # correctly-oriented ground truth images
└── testing/
    ├── input/    # rotated test images
    └── gt/       # correctly-oriented test images
```

Our approach needs correctly-oriented images (it applies its own random rotations during training). Create the flat train/test directories:
```bash
# Test set: copy the correctly-oriented ground truth images (535 images)
cp -r data/datasets/ds_drcd/testing/gt data/datasets/test_drcd

# Training set: copy only the unique source images (1,474 of 5,537 gt images)
# The exact filenames are listed in data/datasets/train_drcd_filenames.txt
mkdir -p data/datasets/train_drcd
while read f; do
  cp "data/datasets/ds_drcd/training/gt/$f" data/datasets/train_drcd/
done < data/datasets/train_drcd_filenames.txt
```

### COCO 2014

1. Download the following from [cocodataset.org](https://cocodataset.org/#download):
   - [2014 Train images](http://images.cocodataset.org/zips/train2014.zip) (83K images, 13GB) — used for training
   - [2014 Val images](http://images.cocodataset.org/zips/val2014.zip) — used to extract the test set

2. The test split labels are included in this repository at `data/datasets/ds_coco/02_coco_val_labels.csv`. This file originates from Fischer et al. and defines which of the 4,536 labeled validation images belong to the test set. Images with label 1 or 2 form the 1,030-image test set; all other labels are excluded.

3. Create the train/test directories:
```bash
# Training: use the full COCO 2014 train set (83K images)
ln -s $(pwd)/data/datasets/ds_coco/train2014 data/datasets/train_coco

# Test: extract the 1,030 test images from val2014 using Fischer's labels
mkdir -p data/datasets/test_coco
python -c "
import pandas as pd, shutil
df = pd.read_csv('data/datasets/ds_coco/02_coco_val_labels.csv')
test = df[df['labels'].isin([1, 2])]
for name in test['image_names']:
    shutil.copy(f'data/datasets/ds_coco/val2014/{name}', 'data/datasets/test_coco/')
print(f'Copied {len(test)} test images')
"
```

### COCO 2017

Download [2017 Train images](http://images.cocodataset.org/zips/train2017.zip) from [cocodataset.org](https://cocodataset.org/#download) and place in `data/datasets/train_coco_2017/`.

Since COCO 2017 shares images with COCO 2014, you must remove test images from the training set to prevent leakage:
```bash
cd data/datasets
bash remove_test_images.sh
```
This moves matching test images out of `train_coco_2017/` into a backup folder. The same COCO 2014 test set (1,030 images from val2014) is used for evaluation.

## Training

### Full comparison (DRC-D)

Reproduces Table 1 from the paper — runs all 5 methods across all 16 architectures with 5 seeds each:

```bash
python compare.py --num-runs 5 --keep-checkpoints best --mixed-precision --batch-size 16
```

### Single model training

```bash
# CGD + MambaOut Base on DRC-D
python train.py --approach cgd --model-name mambaout_base --batch-size 16 --mixed-precision --run-test

# CLS + EfficientViT-B3 on DRC-D
python train.py --approach classification --model-name efficientvit_b3 --batch-size 16 --mixed-precision --run-test

# CGD + MambaOut Base on COCO 2014
python train.py --approach cgd --model-name mambaout_base \
  --train-dir data/datasets/train_coco --validation-split 0.1 \
  --batch-size 16 --mixed-precision \
  --test-dirs data/datasets/test_coco --run-test

# CGD + MambaOut Base on COCO 2017
python train.py --approach cgd --model-name mambaout_base \
  --train-dir data/datasets/train_coco_2017 --validation-split 0.1 \
  --batch-size 16 --mixed-precision \
  --test-dirs data/datasets/test_coco --run-test
```

### Evaluation only

```bash
python train.py --approach cgd --model-name mambaout_base \
  --train-dir data/datasets/train_coco --validation-split 0.1 \
  --batch-size 16 \
  --test-dirs data/datasets/test_coco \
  --test-only --test-ckpt path/to/checkpoint.ckpt \
  --test-random-seed 0
```

Test angles are deterministic for a given seed. Results in the paper use seeds 0–4 and report the mean.

## Demo

```bash
python app.py
```

Launches a Gradio web interface for interactive angle prediction. Upload an image, select a model, and see the predicted rotation angle with a corrected output.

## Hardware

Training was performed on a single NVIDIA GPU. Batch size 16 with mixed precision is recommended. Training time varies by architecture (roughly 1–3 hours per model on DRC-D).

## License

MIT
