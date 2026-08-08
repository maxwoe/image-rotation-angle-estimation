# Image Rotation Angle Estimation

[![arXiv](https://img.shields.io/badge/arXiv-2603.25351-B31C1C?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.25351)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Space-Live%20Demo-blue)](https://huggingface.co/spaces/maxwoe/image-rotation-angle-estimation)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hub-Pretrained%20Models-blue)](https://huggingface.co/maxwoe/image-rotation-angle-estimation)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Estimate how much an image has been rotated from its upright orientation. No labels needed. Includes pretrained models, an interactive demo, and training code for custom datasets.

## Quick Start

Requires Python 3.10+, CUDA 11.8+ recommended. Clone the repo, then:

```bash
pip install -r requirements.txt
```

Download a pretrained checkpoint from the [HuggingFace Hub](https://huggingface.co/maxwoe/image-rotation-angle-estimation) and place it in `weights/`:

```
weights/
└── cgd_mambaout_base_coco2017.ckpt
```

Then launch the demo:

```bash
python app.py  # opens Gradio UI at http://localhost:7861
```

For Python inference, see the [HuggingFace model card](https://huggingface.co/maxwoe/image-rotation-angle-estimation).

## Train on Your Own Images

No annotation files or labels needed. Just a folder of correctly-oriented images. The pipeline applies random rotations during training and learns to predict the applied angle.

```bash
python train.py --approach cgd --model-name mambaout_base \
  --train-dir path/to/your/images --validation-split 0.1 \
  --batch-size 16 --mixed-precision \
  --test-dirs path/to/test/images --run-test
```

### Evaluation only

```bash
python train.py --approach cgd --model-name mambaout_base \
  --train-dir path/to/your/images --validation-split 0.1 \
  --batch-size 16 \
  --test-dirs path/to/test/images \
  --test-only --test-ckpt path/to/checkpoint.ckpt \
  --test-random-seed 0
```

For reproducing paper results on DRC-D and COCO, see [data/DATASETS.md](data/DATASETS.md).

## Results

Best results on DRC-D (mean over five training runs):

| Method | Architecture | MAE (°) |
|---|---|---|
| CLS + CSL labels | MambaOut Tiny | 0.81 |
| CLS | EfficientViT-B3 | 1.23 |
| CGD | MambaOut Base | 1.24 |

Best results on COCO, tested on 1,030 val images (CGD):

| Training Data | Architecture | MAE (°) |
|---|---|---|
| COCO 2014 | ConvNeXt V2 Base | 3.13 |
| COCO 2017 | MambaOut Base | 2.84 |

## Methods

Five circular-aware approaches are implemented and benchmarked; the classification head can additionally be trained with Circular Smooth Labels (`--loss-type csl`), which removes its training instabilities on small datasets and yields the best DRC-D result. CGD never fails to converge and is best on every COCO backbone. See the [paper](https://arxiv.org/abs/2603.25351) for the full comparison.

| Abbreviation | Approach | Description |
|---|---|---|
| DA | Direct Angle | Scalar regression with circular loss |
| CLS | Classification | Angular binning (360 bins), cross-entropy or CSL labels |
| UV | Unit Vector | Regression on (cos θ, sin θ) |
| PSC | Phase-Shifting Coder | Frequency-domain encoding |
| CGD | Circular Gaussian Distribution | Probabilistic soft targets via KL divergence |

## Reproducing the paper's evaluation

- DRC-D grids: `compare.py` (per-method sweeps; `--extra-args` forwards options such as `--validation-split 0.2` or `--no-unit-regularization` to `train.py`).
- COCO cells: train once with `train.py`, then evaluate a checkpoint across five test seeds with `eval/eval_coco_ours.sh` (set `MODELS_FILE=` to a list of `approach|label|checkpoint` lines and `OUTPUT_JSON=` for the output).
- Per-seed result files used in the paper are published: DRC-D under `eval/results/drcd/` (`base_grid_seeds.json` for the five base methods, `csl_sweep_seeds.json` for the CSL variant, and `valsplit20_seeds.json`, `uv_noreg_seeds.json`, `padding_cls_seeds.json`, `padding_cgd_seeds.json` for the ablations), COCO under `eval/results/`. `eval/paper_table.py` and `eval/significance_tests.py` regenerate the DRC-D table rows and the statistical tests from them and run with no arguments.
- Useful `train.py` options added for the revision: `--early-stopping-min-delta` (improvement threshold for early stopping), `--rotation-mode pad` (padding control experiment), `--inference-method` (argmax or soft-argmax decoding), `--no-unit-regularization` (naive two-output regression baseline).

## Citation

```bibtex
@misc{woehrer2026irae,
  title={Image Rotation Angle Estimation: Comparing Circular-Aware Methods},
  author={Woehrer, Maximilian},
  year={2026},
  eprint={2603.25351},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.25351}
}
```
