#!/usr/bin/env bash
# trace_coco_cell.sh
#
# One COCO cell of the paper's large-data grid, at its COCO protocol
# (batch 16, FP32, per-backbone learning rate from architectures.py, one
# training run, five test seeds) under the trace-free protocol
# (--remove-traces), with the early-stopping threshold of 0.1 deg. Afterwards the best checkpoint
# gets the full evaluation protocol: the paper's evaluation path on the test
# views without re-encode and re-encoded once at JPEG q95, q90 and q75 (five
# test seeds each), the JPEG-gap audit with the 400-texture probe (summary row),
# the 1,000-texture certification probe (lossless, q90, q75, q50 textures) and
# the cue-conflict test. Steps are skipped once their marker exists, so the
# script can be re-run to resume. Meant for its own screen session:
#   screen -dmS irae-coco-mambaout bash -c "bash eval/trace_coco_cell.sh mambaout_base.in1k"
#
# usage: bash eval/trace_coco_cell.sh BACKBONE [TRAIN_DIR]
#   BACKBONE   timm name, e.g. mambaout_base.in1k
#   TRAIN_DIR  default data/datasets/train_coco (COCO 2014)
#   EXTRA_ARGS environment, default "--remove-traces"

set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate irae
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BB="${1:?usage: trace_coco_cell.sh BACKBONE [TRAIN_DIR]}"
TRAIN_DIR="${2:-data/datasets/train_coco}"
EXTRA_ARGS="${EXTRA_ARGS:---remove-traces}"
VARIANT="${VARIANT:-camera}"               # names the cell; camera = the trace-free protocol
TEST_BLUR="${TEST_BLUR:-0}"                # test-time blur sigma, for recipes that train with a blur (0 = skip)
ROTATION_MODE="${ROTATION_MODE:-crop}"     # crop (paper), crop_fixed, circle or pad; used for training, evaluation and audit
export ROTATION_MODE                       # read by eval/eval_coco_ours.sh
AUDIT_ROTATION_MODE="$ROTATION_MODE"; export AUDIT_ROTATION_MODE   # read by eval/trace_audit.sh
APPROACH="${APPROACH:-cgd}"                # cgd, classification, unit_vector, psc, direct_angle
LOSS_ARGS="${LOSS_ARGS:-}"                 # e.g. "--loss-type csl" for the CSL variant of classification
SHORT="$(echo "$BB" | cut -d. -f1)"
SET="$(basename "$TRAIN_DIR" | sed 's/^train_//')"          # coco or coco_2017
HEAD="$APPROACH"; [[ "$LOSS_ARGS" == *csl* ]] && HEAD="${APPROACH}_csl"
TAG="${HEAD}_${SHORT}_${SET}_${VARIANT}"
SD="data/saved_models_trace_removal/${TAG}"
AUDIT=eval/results/trace_audit
LISTS=$AUDIT/lists
MARK=$AUDIT/markers
RES=eval/results/trace_removal
LOG=logs/trace_removal
mkdir -p "$SD" "$LISTS" "$MARK" "$RES" "$LOG"

best_ckpt() {  # lowest val_mae_deg among non-last checkpoints under $1
  python - "$1" <<'PY'
import glob, os, re, sys
best, bv = "", 1e9
for f in glob.glob(os.path.join(sys.argv[1], "**", "*.ckpt"), recursive=True):
    if os.path.basename(f) == "last.ckpt":
        continue
    m = re.search(r"val_mae_deg=([0-9.]+)", f)
    if m and float(m.group(1).rstrip(".")) < bv:
        best, bv = f, float(m.group(1).rstrip("."))
print(best)
PY
}

echo "CELL START $TAG $(date '+%F %T')  backbone=$BB train=$TRAIN_DIR extra='$EXTRA_ARGS'"
echo "branch $(git branch --show-current) $(git log --oneline -1)"

if [[ ! -f "$MARK/$TAG.train.done" ]]; then
  echo "======== TRAIN $TAG $(date '+%F %T') ========"
  python train.py --approach "$APPROACH" $LOSS_ARGS --model-name "$BB" \
    --train-dir "$TRAIN_DIR" --batch-size 16 --early-stopping-min-delta 0.1 --rotation-mode "$ROTATION_MODE" \
    --run-test --test-dirs data/datasets/test_coco --save-dir "$SD" $EXTRA_ARGS 2>&1 | tee "$LOG/$TAG.train.log"
  if [[ -n "$(best_ckpt "$SD")" ]]; then
    touch "$MARK/$TAG.train.done"
  else
    echo "TRAIN FAILED: no checkpoint under $SD"; exit 1
  fi
else
  echo "train done earlier, skipping"
fi

CK="$(best_ckpt "$SD")"
echo "best checkpoint: $CK"
printf '%s|%s|%s\n' "$APPROACH" "$TAG" "$CK" > "$LISTS/models_$TAG.txt"

# The four test forms of the benchmark: the rotated crop without re-encode
# ("clean", the published protocol) and re-encoded once at JPEG q95, q90, q75.
for FORM in clean q95 q90 q75; do
  [[ -f "$MARK/$TAG.eval_$FORM.done" ]] && continue
  Q=0; [[ "$FORM" != clean ]] && Q="${FORM#q}"
  SUFFIX=""; [[ "$FORM" != clean ]] && SUFFIX="_$FORM"
  echo "======== EVAL $FORM (TEST_JPEG_QUALITY=$Q), 5 test seeds $(date '+%F %T') ========"
  TEST_JPEG_QUALITY="$Q" COCO_TRAIN_DIR="$TRAIN_DIR" MODELS_FILE="$LISTS/models_$TAG.txt" \
    OUTPUT_JSON="$RES/${TAG}${SUFFIX}_seeds.json" \
    bash eval/eval_coco_ours.sh 2>&1 | grep -v "INFO\|WARNING\|Seed set" | tee "$LOG/$TAG.eval_$FORM.log" \
    && touch "$MARK/$TAG.eval_$FORM.done"
done

if [[ "$TEST_BLUR" != "0" ]]; then
  # matched evaluation: the test views get the same trace removal (blur) the model
  # was trained with, with and without one JPEG save on top
  if [[ ! -f "$MARK/$TAG.eval_blur.done" ]]; then
    echo "======== EVAL blur sigma $TEST_BLUR, 5 test seeds $(date '+%F %T') ========"
    TEST_BLUR_SIGMA="$TEST_BLUR" COCO_TRAIN_DIR="$TRAIN_DIR" MODELS_FILE="$LISTS/models_$TAG.txt" \
      OUTPUT_JSON="$RES/${TAG}_blur${TEST_BLUR}_seeds.json" \
      bash eval/eval_coco_ours.sh 2>&1 | grep -v "INFO\|WARNING\|Seed set" | tee "$LOG/$TAG.eval_blur.log" \
      && touch "$MARK/$TAG.eval_blur.done"
  fi
  if [[ ! -f "$MARK/$TAG.eval_blur_q90.done" ]]; then
    echo "======== EVAL blur sigma $TEST_BLUR + JPEG q90, 5 test seeds $(date '+%F %T') ========"
    TEST_BLUR_SIGMA="$TEST_BLUR" TEST_JPEG_QUALITY=90 COCO_TRAIN_DIR="$TRAIN_DIR" MODELS_FILE="$LISTS/models_$TAG.txt" \
      OUTPUT_JSON="$RES/${TAG}_blur${TEST_BLUR}_q90_seeds.json" \
      bash eval/eval_coco_ours.sh 2>&1 | grep -v "INFO\|WARNING\|Seed set" | tee "$LOG/$TAG.eval_blur_q90.log" \
      && touch "$MARK/$TAG.eval_blur_q90.done"
  fi
fi

echo "======== AUDIT (gap + probe) $(date '+%F %T') ========"
printf 'new_%s|%s|%s|data/datasets/test_coco|0,1,2,3,4|640x480\n' "$TAG" "$APPROACH" "$CK" > "$LISTS/audit_$TAG.txt"
if [[ "$TEST_BLUR" != "0" ]]; then
  CONDS="clean,jpeg90,blur${TEST_BLUR},blur${TEST_BLUR}+jpeg90"
else
  CONDS="clean,jpeg90"
fi
if [[ ! -f "$MARK/$TAG.audit.done" ]]; then
  AUDIT_CONDITIONS="$CONDS" bash eval/trace_audit.sh "$LISTS/audit_$TAG.txt" 2>&1 | tee "$LOG/$TAG.audit.log" \
    && touch "$MARK/$TAG.audit.done"
fi

# Certification: the four gates need the probe at 1,000 textures including the
# strong grids (q75, q50) and the cue-conflict test. grid_probe exits 1 when a
# gate fails; the verdict is in the json, so the exit code is recorded, not acted on.
if [[ ! -f "$MARK/$TAG.certify.done" ]]; then
  echo "======== CERTIFY, 1,000 textures $(date '+%F %T') ========"
  mkdir -p "$AUDIT/certify"
  python eval/grid_probe.py --approach "$APPROACH" --ckpt "$CK" --n 1000 --rotation-mode "$ROTATION_MODE" \
    --out "$AUDIT/certify/${TAG}_n1000.json" 2>&1 | tee "$LOG/$TAG.certify.log"
  echo "grid_probe exit code ${PIPESTATUS[0]} (1 = a gate failed, see the json)" | tee -a "$LOG/$TAG.certify.log"
  [[ -f "$AUDIT/certify/${TAG}_n1000.json" ]] && touch "$MARK/$TAG.certify.done"
fi
if [[ ! -f "$MARK/$TAG.cue_conflict.done" ]]; then
  echo "======== CUE CONFLICT $(date '+%F %T') ========"
  mkdir -p "$AUDIT/cue_conflict"
  python eval/cue_conflict.py --approach "$APPROACH" --ckpt "$CK" --n 300 \
    --out "$AUDIT/cue_conflict/${TAG}.json" 2>&1 | tee "$LOG/$TAG.cue_conflict.log" \
    && touch "$MARK/$TAG.cue_conflict.done"
fi

echo "CELL_DONE $TAG $(date '+%F %T')"
