#!/usr/bin/env bash
# trace_pipeline.sh
#
# End-to-end trace-removal study on the A100, meant to run inside a screen
# session (screen -S irae-trace). Steps, each skipped when its DONE marker
# exists so the script can be re-run to resume:
#
#   1. audit_paper_drcd   JPEG gap + probe on the paper's best DRC-D CSL and CGD checkpoints
#   2. audit_paper_coco   JPEG gap + probe on the paper's COCO CGD cells
#   3. retrain_cgd        CGD on all 16 DRC-D backbones, 5 runs, --remove-traces
#   4. audit_new_cgd      gap + probe on the new CGD checkpoints
#   5. retrain_csl        CLS (CSL) on all 16 backbones, 5 runs, --remove-traces
#   6. audit_new_csl      gap + probe on the new CSL checkpoints
#
# Everything is logged under logs/trace_removal/. Summary table:
# eval/results/trace_audit/summary.tsv. Run from the repository root:
#   bash eval/trace_pipeline.sh

set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate irae
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG_DIR=logs/trace_removal
AUDIT=eval/results/trace_audit
LISTS=$AUDIT/lists
MARK=$AUDIT/markers
mkdir -p "$LOG_DIR" "$LISTS" "$MARK"

MODELS="convnextv2_atto.fcmae_ft_in1k convnextv2_base.fcmae_ft_in22k_in1k edgenext_xx_small.in1k edgenext_base.in21k_ft_in1k efficientnetv2_rw_t.ra2_in1k efficientnetv2_rw_m.agc_in1k efficientvit_b0.r224_in1k efficientvit_b3.r224_in1k focalnet_tiny_lrf.ms_in1k focalnet_base_lrf.ms_in1k mambaout_tiny.in1k mambaout_base.in1k swin_tiny_patch4_window7_224 swin_base_patch4_window7_224 vit_tiny_patch16_224.augreg_in21k_ft_in1k vit_base_patch16_224.augreg_in21k_ft_in1k"
# the first recipe of the study (blur and partial re-encoding); the paper's trace-free
# protocol is plain --remove-traces
EXTRA_ARGS="${EXTRA_ARGS:---remove-traces --remove-traces-blur 2.5,4.0 --remove-traces-jpeg 30,95 --remove-traces-jpeg-prob 0.6}"
TAG="${TAG:-trace_removal}"

step() {  # step NAME COMMAND...
  local name=$1; shift
  if [[ -f "$MARK/$name.done" ]]; then
    echo "[$name] done earlier, skipping"
    return 0
  fi
  echo ""
  echo "########## [$name] START $(date '+%F %T')"
  nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
  if "$@" 2>&1 | tee "$LOG_DIR/$name.log"; then
    touch "$MARK/$name.done"
    echo "########## [$name] DONE $(date '+%F %T')"
  else
    echo "########## [$name] FAILED $(date '+%F %T') (see $LOG_DIR/$name.log)"
    return 1
  fi
}

retrain() {  # retrain APPROACH OUTPUT_JSON
  python compare.py --approaches "$1" --models $MODELS --num-runs 5 --mixed-precision \
    --keep-checkpoints best --extra-args "$EXTRA_ARGS" --output "$2"
}

audit_new() {  # audit_new STAMP LISTNAME
  python eval/trace_audit_lists.py --new-since "$1" --out "$LISTS/$2.txt" && bash eval/trace_audit.sh "$LISTS/$2.txt"
}

echo "PIPELINE START $(date '+%F %T') on $(hostname), branch $(git branch --show-current) $(git log --oneline -1)"
echo "extra train args: $EXTRA_ARGS"

python eval/trace_audit_lists.py --paper
step audit_paper_drcd bash eval/trace_audit.sh "$LISTS/paper_drcd.txt" || exit 1
step audit_paper_coco bash eval/trace_audit.sh "$LISTS/paper_coco.txt" || exit 1

[[ -f "$MARK/stamp_cgd" ]] || touch "$MARK/stamp_cgd"
step retrain_cgd retrain cgd "cgd_${TAG}_results.json" || exit 1
step audit_new_cgd audit_new "$MARK/stamp_cgd" "new_cgd_${TAG}" || exit 1

[[ -f "$MARK/stamp_csl" ]] || touch "$MARK/stamp_csl"
step retrain_csl retrain classification_csl "csl_${TAG}_results.json" || exit 1
step audit_new_csl audit_new "$MARK/stamp_csl" "new_csl_${TAG}" || exit 1

echo "PIPELINE_DONE $(date '+%F %T')"
