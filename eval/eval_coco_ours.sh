#!/usr/bin/env bash
# eval_coco_ours.sh
#
# Evaluates the 4 "Ours" models from Table 2 on the COCO test sets
# (all, easy, hard) with 5 independent test seeds (0-4), collecting
# all results into a single JSON file.
#
# Run from the repo root:
#   bash eval/eval_coco_ours.sh
#
# Optional overrides:
#   COCO_TRAIN_DIR  (default: data/datasets/train_coco)
#   OUTPUT_JSON     (default: eval/results/coco_ours_seeds.json)

set -euo pipefail

COCO_TRAIN_DIR="${COCO_TRAIN_DIR:-data/datasets/train_coco}"
OUTPUT_JSON="${OUTPUT_JSON:-eval/results/coco_ours_seeds.json}"

SEEDS=(0 1 2 3 4)

declare -a SUBSETS=(
  "all|data/datasets/test_coco"
  #"easy|data/datasets/test_coco_easy"
  #"hard|data/datasets/test_coco_hard"
)

# "approach|model_name|checkpoint_relative_path"
declare -a MODELS=(
  "direct_angle|convnextv2_atto-32|data/saved_models/direct_angle-mae-convnextv2_atto-32/version_0/checkpoints/direct_angle-mae-convnextv2_atto-32-version=0-epoch=0050-step=0237507-train_loss=21.2573-val_loss=20.8054-train_mae_deg_epoch=21.2573-val_mae_deg=20.8054.ckpt"
  "classification|efficientvit_b3|data/saved_models/classification-cross_entropy-efficientvit_b3-32/version_0/checkpoints/classification-cross_entropy-efficientvit_b3-32-version=1-epoch=0059-step=0279420-train_loss=1.7354-val_loss=1.8317-train_mae_deg_epoch=4.9430-val_mae_deg=7.3192.ckpt"
  "unit_vector|convnextv2_base|data/saved_models/unit_vector-mae-convnextv2_base-32/version_0/checkpoints/unit_vector-mae-convnextv2_base-32-version=0-epoch=0072-step=0339961-train_loss=0.0647-val_loss=0.0730-train_mae_deg_epoch=5.6013-val_mae_deg=7.3988.ckpt"
  "psc|convnextv2_base|data/saved_models/psc-mae-convnextv2_base-32/version_0/checkpoints/psc-mae-convnextv2_base-32-version=0-epoch=0086-step=0405159-train_loss=0.0678-val_loss=0.0785-train_mae_deg_epoch=5.8839-val_mae_deg=8.0302.ckpt"
  "cgd|mambaout_base|data/saved_models/cgd-kl_divergence-mambaout_base-32/version_0/checkpoints/cgd-kl_divergence-mambaout_base-32-version=0-epoch=0051-step=0242164-train_loss=0.0274-val_loss=0.2393-train_mae_deg_epoch=1.0981-val_mae_deg=6.1074.ckpt"
  "cgd|mambaout_base|data/saved_models/cgd-kl_divergence-mambaout_base-32/version_1/checkpoints/cgd-kl_divergence-mambaout_base-32-version=1-epoch=0070-step=0468813-train_loss=0.0169-val_loss=0.2808-train_mae_deg_epoch=0.8398-val_mae_deg=6.2294.ckpt"
)
# last entry is trained on coco 2017

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
for subset_def in "${SUBSETS[@]}"; do
  IFS='|' read -r subset_name test_dir <<< "$subset_def"
  if [[ ! -d "$test_dir" ]]; then
    echo "ERROR: Test directory not found: $test_dir"
    exit 1
  fi
done

mkdir -p "$(dirname "$OUTPUT_JSON")"

# ---------------------------------------------------------------------------
# Run + collect
# ---------------------------------------------------------------------------
TMP="$(mktemp)"
echo "[" > "$TMP"
first_entry=true

total=$(( ${#SUBSETS[@]} * ${#MODELS[@]} * ${#SEEDS[@]} ))
count=0

for subset_def in "${SUBSETS[@]}"; do
  IFS='|' read -r subset_name test_dir <<< "$subset_def"

  for model_def in "${MODELS[@]}"; do
    IFS='|' read -r approach model_name ckpt <<< "$model_def"

    for seed in "${SEEDS[@]}"; do
      count=$(( count + 1 ))
      echo ""
      echo "=== [$count/$total] subset=$subset_name  approach=$approach  model=$model_name  seed=$seed ==="

      output="$(python train.py \
        --approach "$approach" \
        --model-name "$model_name" \
        --train-dir "$COCO_TRAIN_DIR" \
        --validation-split 0.1 \
        --batch-size 16 \
        --test-dirs "$test_dir" \
        --test-only \
        --test-ckpt "$ckpt" \
        --test-random-seed "$seed" \
        2>&1)"

      echo "$output"

      # Extract key=value pairs between the sentinel lines
      metrics_block="$(echo "$output" \
        | awk '/=== TEST_RESULTS_START ===/,/=== TEST_RESULTS_END ===/' \
        | grep -v 'TEST_RESULTS_')"

      # Append JSON entry
      if [[ "$first_entry" == "true" ]]; then
        first_entry=false
      else
        echo "," >> "$TMP"
      fi

      {
        printf '  {\n'
        printf '    "subset": "%s",\n' "$subset_name"
        printf '    "approach": "%s",\n' "$approach"
        printf '    "model_name": "%s",\n' "$model_name"
        printf '    "seed": %d,\n' "$seed"
        printf '    "metrics": {'
        first_metric=true
        while IFS='=' read -r key val; do
          [[ -z "$key" ]] && continue
          if [[ "$first_metric" == "true" ]]; then
            first_metric=false
            printf '\n'
          else
            printf ',\n'
          fi
          printf '      "%s": %s' "$key" "$val"
        done <<< "$metrics_block"
        printf '\n    }\n'
        printf '  }'
      } >> "$TMP"
    done
  done
done

echo "" >> "$TMP"
echo "]" >> "$TMP"

mv "$TMP" "$OUTPUT_JSON"
echo ""
echo "=== Done. Results saved to: $OUTPUT_JSON ==="
