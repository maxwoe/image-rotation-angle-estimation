#!/usr/bin/env bash
# trace_stage1_drcd.sh
#
# Stage 1 of the trace-removal study on DRC-D (cheap screening), in a screen
# session (screen -S irae-stage1). Steps are skipped once their marker exists.
#
#   0. make_q75          DRC-D re-encoded at JPEG quality 75 (strong source grid)
#   1. testbed_q75       plain CGD on the q75 sets, 2 runs x 2 backbones: does a small-data
#                        model learn a strong grid? (probe + gap decide whether DRC-D can
#                        screen for shortcut removal, not only for cost)
#   2. recipes           CGD on DRC-D, 5 runs x 2 backbones per recipe, tested on q90 views
#                        with matched preprocessing; then gap + probe on the best run.
#
# Recipes (train.py extra args; the in-run test uses --test-jpeg-quality 90 and, for blur
# recipes, --test-blur-sigma at the middle of the training range):
#   r0_baseline   paper protocol
#   r1_camera     JPEG re-encode of every view, q40-90, no blur
#   r2_blur1      blur sigma 1.0-1.5 + camera JPEG
#   r3_blur2      blur sigma 1.5-2.5 + camera JPEG
#   r4_blur3      default trace removal (blur 2.5-4.0, JPEG q30-95 p0.6)
#   r5_fixedcrop  camera JPEG + angle-independent crop
#   r6_scale      camera JPEG + random downscale 0.5-1.0
#   r7_camera_q85 every view re-encoded at a fixed q85 (control for r1: codec after the
#                 geometry without stochastic JPEG augmentation)
#
# Results: comparison/stage1_<recipe>_results.json (5-run means), audit rows in
# eval/results/trace_audit/summary.tsv (labels stage1_<recipe>_*), logs in logs/trace_stage1/.

set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate irae
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODELS="${MODELS:-mambaout_tiny.in1k convnextv2_base.fcmae_ft_in22k_in1k}"
RUNS="${RUNS:-5}"
LOG=logs/trace_stage1
AUDIT=eval/results/trace_audit
LISTS=$AUDIT/lists
MARK=$AUDIT/markers
mkdir -p "$LOG" "$LISTS" "$MARK"

step() {  # step NAME COMMAND...
  local name=$1; shift
  if [[ -f "$MARK/stage1_$name.done" ]]; then echo "[$name] done earlier, skipping"; return 0; fi
  echo ""; echo "########## [$name] START $(date '+%F %T')"
  if "$@" 2>&1 | tee "$LOG/$name.log"; then
    touch "$MARK/stage1_$name.done"; echo "########## [$name] DONE $(date '+%F %T')"
  else
    echo "########## [$name] FAILED $(date '+%F %T')"; return 1
  fi
}

audit_new() {  # audit_new STAMP LISTNAME TEST_DIR CONDITIONS
  python eval/trace_audit_lists.py --new-since "$1" --out "$LISTS/$2.txt" --test-dir "$3" \
    && AUDIT_CONDITIONS="$4" bash eval/trace_audit.sh "$LISTS/$2.txt"
}

recipe() {  # recipe NAME TRAIN_EXTRA_ARGS AUDIT_CONDITIONS [ROTATION_MODE]
  local name=$1 extra=$2 conds=$3 rmode=${4:-crop}
  [[ -f "$MARK/stage1_stamp_$name" ]] || touch "$MARK/stage1_stamp_$name"
  step "train_$name" python compare.py --approaches cgd --models $MODELS --num-runs "$RUNS" --mixed-precision \
    --keep-checkpoints best --extra-args "$extra" --output "stage1_${name}_results.json" || return 1
  AUDIT_ROTATION_MODE="$rmode" step "audit_$name" audit_new "$MARK/stage1_stamp_$name" "stage1_$name" \
    data/datasets/test_drcd "$conds" || return 1
}

echo "STAGE1 START $(date '+%F %T') branch $(git branch --show-current) $(git log --oneline -1)"
echo "models: $MODELS, runs: $RUNS"

# 0 + 1: the quality-75 testbed
step make_q75 python eval/make_drcd_q75.py 75 || exit 1
[[ -f "$MARK/stage1_stamp_q75" ]] || touch "$MARK/stage1_stamp_q75"
step testbed_q75_train python compare.py --approaches cgd --models $MODELS --num-runs 2 --mixed-precision \
  --keep-checkpoints best --train-dir data/datasets/train_drcd_q75 --test-dirs data/datasets/test_drcd_q75 \
  --output stage1_q75_baseline_results.json || exit 1
step testbed_q75_audit audit_new "$MARK/stage1_stamp_q75" stage1_q75_baseline data/datasets/test_drcd_q75 clean,jpeg90 || exit 1

# 2: recipes on the normal DRC-D sets
CAM="--remove-traces --remove-traces-blur 0,0 --remove-traces-jpeg 40,90 --remove-traces-jpeg-prob 1.0"
recipe r0_baseline  "--test-jpeg-quality 90"                                                   "clean,jpeg90"
recipe r1_camera    "$CAM --test-jpeg-quality 90"                                              "clean,jpeg90"
recipe r2_blur1     "$CAM --remove-traces-blur 1.0,1.5 --test-blur-sigma 1.25 --test-jpeg-quality 90" "clean,jpeg90,blur1.25,blur1.25+jpeg90"
recipe r3_blur2     "$CAM --remove-traces-blur 1.5,2.5 --test-blur-sigma 2.0 --test-jpeg-quality 90"  "clean,jpeg90,blur2,blur2+jpeg90"
recipe r4_blur3     "--remove-traces --remove-traces-blur 2.5,4.0 --remove-traces-jpeg 30,95 --remove-traces-jpeg-prob 0.6 --test-blur-sigma 3 --test-jpeg-quality 90" "clean,jpeg90,blur3,blur3+jpeg90"
recipe r5_fixedcrop "$CAM --rotation-mode crop_fixed --test-jpeg-quality 90"                   "clean,jpeg90" crop_fixed
recipe r6_scale     "$CAM --remove-traces-scale 0.5,1.0 --test-jpeg-quality 90"                "clean,jpeg90"
# control for r1: a fixed-quality final encode separates "codec after the geometry" from
# stochastic JPEG augmentation
recipe r7_camera_q85 "--remove-traces --remove-traces-blur 0,0 --remove-traces-jpeg 85,85 --remove-traces-jpeg-prob 1.0 --test-jpeg-quality 90" "clean,jpeg90"

echo "STAGE1_DONE $(date '+%F %T')"
