# Image Rotation Angle Estimation

Deep learning for predicting image rotation angles using PyTorch Lightning. Supports multiple approaches: direct angle regression with circular loss (DA), classification via angular binning (CLS), unit vector regression (UV), phase-shifting coder (PSC), and circular Gaussian distribution (CGD). We test these methods across sixteen modern architectures to identify the best combinations and provide practical guidance for method selection.

## Quick Start

```bash
# Train with unit vector approach
python train.py --approach unit_vector

# Run interactive demo
python app.py
```

## Train
```bash
# Comparison on DRC-D
python compare.py --num-runs 5 --keep-checkpoints best --mixed-precision --batch-size 16

# Train best combo on COCO
python train.py --approach direct_angle --model-name convnextv2_atto --train-dir data/datasets/train_coco --validation-split 0.1 --batch-size 16 --test-dirs data/datasets/test_coco --run-test

python train.py --approach classification --model-name efficientvit_b3 --train-dir data/datasets/train_coco --validation-split 0.1 --test-dirs data/datasets/test_coco --run-test

python train.py --approach unit_vector --model-name convnextv2_base --train-dir data/datasets/train_coco --validation-split 0.1 --batch-size 16 --test-dirs data/datasets/test_coco --run-test

python train.py --approach psc --model-name convnextv2_base --train-dir data/datasets/train_coco --validation-split 0.1 --batch-size 16 --test-dirs data/datasets/test_coco --run-test

python train.py --approach cgd --model-name mambaout_base --train-dir data/datasets/train_coco --validation-split 0.1 --batch-size 16 --test-dirs data/datasets/test_coco --run-test

# Train overall best combo COCO 2017
python train.py --approach cgd --model-name mambaout_base --train-dir data/datasets/train_coco_2017 --validation-split 0.1 --batch-size 16 --test-dirs data/datasets/test_coco --run-test

# Test model
python train.py --approach cgd --model-name mambaout_base --train-dir data/datasets/train_coco --validation-split 0.1 --batch-size 16 --test-dirs data/datasets/test_coco --test-only --test-ckpt data/saved_models/cgd-kl_divergence-mambaout_base-32/version_0/checkpoints/cgd-kl_divergence-mambaout_base-32-version=0-epoch=0051-step=0242164-train_loss=0.0274-val_loss=0.2393-train_mae_deg_epoch=1.0981-val_mae_deg=6.1074.ckpt --test-random-seed 0
```

# Models
/direct_angle-mae-convnextv2_atto-32/version_0/checkpoints/direct_angle-mae-convnextv2_atto-32-version=0-epoch=0050-step=0237507-train_loss=21.2573-val_loss=20.8054-train_mae_deg_epoch=21.2573-val_mae_deg=20.8054.ckpt

data/saved_models/classification-cross_entropy-efficientvit_b3-32/version_0/checkpoints/classification-cross_entropy-efficientvit_b3-32-version=1-epoch=0059-step=0279420-train_loss=1.7354-val_loss=1.8317-train_mae_deg_epoch=4.9430-val_mae_deg=7.3192.ckpt

data/saved_models/unit_vector-mae-convnextv2_base-32/version_0/checkpoints/unit_vector-mae-convnextv2_base-32-version=0-epoch=0072-step=0339961-train_loss=0.0647-val_loss=0.0730-train_mae_deg_epoch=5.6013-val_mae_deg=7.3988.ckpt

data/saved_models/psc-mae-convnextv2_base-32/version_0/checkpoints/psc-mae-convnextv2_base-32-version=0-epoch=0086-step=0405159-train_loss=0.0678-val_loss=0.0785-train_mae_deg_epoch=5.8839-val_mae_deg=8.0302.ckpt

data/saved_models/cgd-kl_divergence-mambaout_base-32/version_0/checkpoints/cgd-kl_divergence-mambaout_base-32-version=0-epoch=0051-step=0242164-train_loss=0.0274-val_loss=0.2393-train_mae_deg_epoch=1.0981-val_mae_deg=6.1074.ckpt