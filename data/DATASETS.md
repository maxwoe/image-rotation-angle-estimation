# Dataset Setup

Only needed to reproduce the paper results. For training on your own images, see the main [README](../README.md).


## DRC-D

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

Create the flat train/test directories:
```bash
# Test set: copy the correctly-oriented ground truth images (535 images)
cp -r data/datasets/ds_drcd/testing/gt data/datasets/test_drcd

# Training set: copy only the unique source images (1,474 of 5,537 gt images)
mkdir -p data/datasets/train_drcd
while read f; do
  cp "data/datasets/ds_drcd/training/gt/$f" data/datasets/train_drcd/
done < data/datasets/train_drcd_filenames.txt
```

## COCO 2014

1. Download from [cocodataset.org](https://cocodataset.org/#download):
   - [2014 Train images](http://images.cocodataset.org/zips/train2014.zip) (83K images, 13GB, for training)
   - [2014 Val images](http://images.cocodataset.org/zips/val2014.zip) (for test set extraction)

2. The test split labels are in `data/datasets/ds_coco/02_coco_val_labels.csv` (from Fischer et al.). Images with label 1 or 2 form the 1,030-image test set.

3. Create the train/test directories:
```bash
ln -s $(pwd)/data/datasets/ds_coco/train2014 data/datasets/train_coco

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

## COCO 2017

Download [2017 Train images](http://images.cocodataset.org/zips/train2017.zip) and place in `data/datasets/train_coco_2017/`.

Since COCO 2017 shares images with COCO 2014, remove test images to prevent leakage:
```bash
cd data/datasets
bash remove_test_images.sh
```

The same COCO 2014 test set (1,030 images) is used for evaluation.

## Training on Paper Datasets

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

### Full benchmark (reproduces paper Table 1)

Runs all 5 methods across all 16 architectures with 5 seeds each:

```bash
python compare.py --num-runs 5 --keep-checkpoints best --mixed-precision --batch-size 16
```
