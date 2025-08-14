Usage Examples:

  1. Resume Full Training (preserves LR scheduler state - may have low LR):
  python train.py --approach unit_vector --resume-ckpt path/to/checkpoint.ckpt

  2. Load Pretrained Weights Only (fresh optimizer/scheduler with your specified LR):
  python train.py --approach unit_vector --pretrained-ckpt path/to/checkpoint.ckpt --learning-rate 0.01

  Key Differences:

  | Option            | Model Weights | Optimizer State | LR Scheduler | Epoch Counter  | Use Case                          |
  |-------------------|---------------|-----------------|--------------|----------------|-----------------------------------|
  | --resume-ckpt     | ✅ Restored    | ✅ Restored      | ✅ Restored   | ✅ Continues    | Continue interrupted training     |
  | --pretrained-ckpt | ✅ Loaded      | 🆕 Fresh        | 🆕 Fresh     | 🆕 Starts at 0 | Fine-tune with different settings |

  The --pretrained-ckpt option is perfect when you want to:
  - Use trained weights as a starting point
  - Apply different learning rates or optimizers
  - Start fresh training dynamics
  - Avoid the "low LR" problem you encountered

  This gives you the best of both worlds - pretrained model knowledge with fresh training dynamics!