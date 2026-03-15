#!/usr/bin/env bash
# eval_coco_oad360.sh
#
# Evaluates OAD-360 on the COCO test set with 5 independent test seeds (0-4),
# collecting all results into a single JSON file.
#
# Requires the `dioad` conda env (OAD-360 deps) to be active.
#
# Run from anywhere:
#   conda activate dioad
#   bash /path/to/image-rotation-angle-estimation/eval/eval_coco_oad360.sh
#
# Required:
#   DIOAD_REPO   path to deep-image-orientation-angle-detection repo
#                (default: sibling directory ../deep-image-orientation-angle-detection)
#
# Optional overrides:
#   COCO_TEST_DIR  (default: <this repo>/data/datasets/test_coco)
#   MODEL_PATH     (default: <DIOAD_REPO>/weights/model-vit-ang-loss.h5)
#   OUTPUT_JSON    (default: <this repo>/eval/results/oad360_seeds.json)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

DIOAD_REPO="${DIOAD_REPO:-$(dirname "$REPO_ROOT")/deep-image-orientation-angle-detection}"
COCO_TEST_DIR="${COCO_TEST_DIR:-$REPO_ROOT/data/datasets/test_coco}"
MODEL_PATH="${MODEL_PATH:-$DIOAD_REPO/weights/model-vit-ang-loss.h5}"
OUTPUT_JSON="${OUTPUT_JSON:-$SCRIPT_DIR/results/oad360_seeds.json}"

TEST_PY="$SCRIPT_DIR/oad360/test_oad360.py"

if [[ ! -d "$DIOAD_REPO" ]]; then
  echo "ERROR: DIOAD repo not found: $DIOAD_REPO"
  echo "       Set DIOAD_REPO=/path/to/deep-image-orientation-angle-detection"
  exit 1
fi

SEEDS=(0 1 2 3 4)

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [[ ! -d "$COCO_TEST_DIR" ]]; then
  echo "ERROR: COCO test directory not found: $COCO_TEST_DIR"
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: OAD-360 model not found: $MODEL_PATH"
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_JSON")"

# ---------------------------------------------------------------------------
# Run + collect
# ---------------------------------------------------------------------------
TMP="$(mktemp)"
echo "[" > "$TMP"
first_entry=true

total=${#SEEDS[@]}
count=0

for seed in "${SEEDS[@]}"; do
  count=$(( count + 1 ))
  echo ""
  echo "=== [$count/$total] OAD-360  seed=$seed ==="

  exit_code=0
  output="$(cd "$DIOAD_REPO" && TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH="$DIOAD_REPO" python "$TEST_PY" \
    --test-dir "$COCO_TEST_DIR" \
    --model-path "$MODEL_PATH" \
    --seed "$seed" \
    2>&1)" || exit_code=$?

  echo "$output"

  if [[ $exit_code -ne 0 ]]; then
    echo "ERROR: Python exited with code $exit_code — skipping seed $seed"
    continue
  fi

  # Extract key=value pairs between the sentinel lines
  metrics_block="$(echo "$output" \
    | awk '/=== TEST_RESULTS_START ===/,/=== TEST_RESULTS_END ===/' \
    | grep -v 'TEST_RESULTS_' || true)"

  if [[ "$first_entry" == "true" ]]; then
    first_entry=false
  else
    echo "," >> "$TMP"
  fi

  {
    printf '  {\n'
    printf '    "approach": "oad360",\n'
    printf '    "model_name": "vit-ang-loss",\n'
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

echo "" >> "$TMP"
echo "]" >> "$TMP"

mv "$TMP" "$OUTPUT_JSON"
echo ""
echo "=== Done. Results saved to: $OUTPUT_JSON ==="
