#!/usr/bin/env bash
# trace_matrix.sh
#
# One worker of the paper's COCO grid: trains the given cells on one backbone
# under the trace-free protocol (JPEG re-encode of every view after rotation
# and crop at a random quality 40-90, random resampler, no blur, no downscale;
# the defaults of --remove-traces, spelled out below),
# one cell after the other, each through trace_coco_cell.sh (train, four-form
# evaluation, audit, certification, cue conflict).
#
# usage: bash eval/trace_matrix.sh WORKER BACKBONE "cell cell ..."
#   cell = head[@2017][@strict]   head in cgd csl ce unit_vector psc direct_angle
#   @2017   trains on COCO 2017 (data/datasets/train_coco_2017) instead of COCO 2014
#   @strict adds the random downscale 0.5-1.0 (the strict variant, for plain cross-entropy)
#   @seedN  training replicate with --random-seed N (default seed 42; the cell tag gets the suffix _seedN)
#
# The matrix of 11 Oct 2026, four workers in parallel on the A100:
#   screen -dmS irae-w1-effvit   bash -c "bash eval/trace_matrix.sh w1 efficientvit_b3.r224_in1k 'cgd csl ce unit_vector psc direct_angle' > logs/trace_removal/worker_w1.log 2>&1"
#   screen -dmS irae-w2-mambaout bash -c "bash eval/trace_matrix.sh w2 mambaout_base.in1k 'cgd csl ce unit_vector psc direct_angle' > logs/trace_removal/worker_w2.log 2>&1"
#   screen -dmS irae-w3-convnext bash -c "bash eval/trace_matrix.sh w3 convnextv2_base.fcmae_ft_in22k_in1k 'cgd csl ce unit_vector psc direct_angle' > logs/trace_removal/worker_w3.log 2>&1"
#   screen -dmS irae-w4-final    bash -c "bash eval/trace_matrix.sh w4 convnextv2_base.fcmae_ft_in22k_in1k 'cgd@2017 ce@strict' > logs/trace_removal/worker_w4.log 2>&1"
#
# Training replicates of the two headline cells, 23 Oct 2026 (seeds 1 and 2 next to the default 42):
#   screen -dmS irae-r1 bash -c "bash eval/trace_matrix.sh r1 convnextv2_base.fcmae_ft_in22k_in1k 'cgd@seed1' > logs/trace_removal/worker_r1.log 2>&1"
#   (r2: cgd@seed2, r3: cgd@2017@seed1, r4: cgd@2017@seed2)

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
W="${1:?usage: trace_matrix.sh WORKER BACKBONE \"cell cell ...\"}"
BB="${2:?backbone}"
CELLS="${3:?cells}"
CAMERA="--remove-traces --remove-traces-blur 0,0 --remove-traces-jpeg 40,90 --remove-traces-jpeg-prob 1.0"

for CELL in $CELLS; do
  HEAD="${CELL%%@*}"
  OPTS="${CELL#"$HEAD"}"
  export EXTRA_ARGS="$CAMERA" VARIANT=camera TEST_BLUR=0 ROTATION_MODE=crop LOSS_ARGS=""
  TRAIN_DIR=data/datasets/train_coco
  [[ "$OPTS" == *@2017* ]] && TRAIN_DIR=data/datasets/train_coco_2017
  [[ "$OPTS" == *@strict* ]] && export EXTRA_ARGS="$CAMERA --remove-traces-scale 0.5,1.0" VARIANT=camera_scale
  if [[ "$OPTS" =~ @seed([0-9]+) ]]; then
    export EXTRA_ARGS="$EXTRA_ARGS --random-seed ${BASH_REMATCH[1]}" VARIANT="${VARIANT}_seed${BASH_REMATCH[1]}"
  fi
  case "$HEAD" in
    csl) export APPROACH=classification LOSS_ARGS="--loss-type csl" ;;
    ce)  export APPROACH=classification ;;
    cgd|unit_vector|psc|direct_angle) export APPROACH="$HEAD" ;;
    *) echo "WORKER $W: unknown head '$HEAD' in cell '$CELL'"; continue ;;
  esac
  echo "WORKER $W: $CELL on $BB ($TRAIN_DIR, $VARIANT)  $(date '+%F %T')"
  bash eval/trace_coco_cell.sh "$BB" "$TRAIN_DIR" || echo "WORKER $W: $CELL on $BB FAILED $(date '+%F %T')"
done
echo "WORKER $W DONE $(date '+%F %T')"
