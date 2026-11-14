#!/usr/bin/env bash
# trace_final_batch.sh
#
# The last evaluation batch for the revision (evaluation only, no training):
#   published_q90    the published Table 2 checkpoints (18 COCO 2014 cells, 3 COCO 2017 cells)
#                    on q90 views through the paper's evaluation path, for the "as published" table
#   quality_curves   JPEG q95..q50 and WebP q90 curves for the CGD cells (three backbones, seed 42),
#                    the release model and the published EfficientViT cell
#   quality_curves_pinned BACKBONE   the same curves with the paper's image-angle pairing, for the figure
#   residue GROUP    the rotation-residue control (eval/rotation_residue.py) on the trace-removal
#                    cells whose tag matches GROUP
#   figures          error histogram and qualitative figure with the release model
#
#   screen -dmS irae-fb-q90 bash -c "bash eval/trace_final_batch.sh published_q90 > logs/trace_removal/final_published_q90.log 2>&1"

set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate irae
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${1:?usage: trace_final_batch.sh published_q90 | quality_curves | quality_curves_pinned BACKBONE | residue GROUP | figures}"
LISTS=eval/results/trace_audit/lists
RES=eval/results/trace_removal
AUD=eval/results/trace_audit
RELEASE_TAG=cgd_convnextv2_base_coco_2017_camera_seed2
ck_of() { cut -d'|' -f3 "$LISTS/models_$1.txt"; }

case "$MODE" in
  published_q90)
    head -18 eval/matrix_models_all.txt > "$LISTS/published_coco2014.txt"
    { cat eval/coco2017_all_models.txt; grep -o '"cgd|mambaout_base|[^"]*version_1[^"]*"' eval/eval_coco_ours.sh | tr -d '"' \
        | sed 's/^cgd|mambaout_base|/cgd|mambaout_base_coco2017|/'; } | grep -v '^#' | grep . > "$LISTS/published_coco2017.txt"
    echo "published: $(wc -l < "$LISTS/published_coco2014.txt") COCO 2014 + $(wc -l < "$LISTS/published_coco2017.txt") COCO 2017 checkpoints"
    TEST_JPEG_QUALITY=90 MODELS_FILE="$LISTS/published_coco2014.txt" OUTPUT_JSON="$RES/published_coco2014_q90_seeds.json" \
      bash eval/eval_coco_ours.sh 2>&1 | grep -v "INFO\|WARNING\|Seed set"
    TEST_JPEG_QUALITY=90 COCO_TRAIN_DIR=data/datasets/train_coco_2017 MODELS_FILE="$LISTS/published_coco2017.txt" \
      OUTPUT_JSON="$RES/published_coco2017_q90_seeds.json" bash eval/eval_coco_ours.sh 2>&1 | grep -v "INFO\|WARNING\|Seed set"
    ;;
  quality_curves)
    CONDS="clean,jpeg95,jpeg90,jpeg80,jpeg70,jpeg60,jpeg50,webp90"
    for pair in "camera_effvit:cgd_efficientvit_b3_coco_camera" "camera_mambaout:cgd_mambaout_base_coco_camera" \
                "camera_convnext:cgd_convnextv2_base_coco_camera" "release_convnext_coco2017:$RELEASE_TAG"; do
      name="${pair%%:*}"; tag="${pair#*:}"
      [[ -f "$AUD/quality_curve/$name.json" ]] && continue
      echo "======== CURVE $name ($tag) $(date '+%F %T') ========"
      python eval/jpeg_gap.py --approach cgd --ckpt "$(ck_of "$tag")" --seeds 0,1,2,3,4 --conditions "$CONDS" \
        --out "$AUD/quality_curve/$name.json" 2>&1 | grep -v "INFO\|Warning"
    done
    if [[ ! -f "$AUD/quality_curve/paper_effvit.json" ]]; then
      echo "======== CURVE paper_effvit $(date '+%F %T') ========"
      python eval/jpeg_gap.py --approach cgd --ckpt "$(grep '^cgd|efficientvit_b3|' eval/matrix_models_all.txt | cut -d'|' -f3)" \
        --seeds 0,1,2,3,4 --conditions "$CONDS" --out "$AUD/quality_curve/paper_effvit.json" 2>&1 | grep -v "INFO\|Warning"
    fi
    ;;
  quality_curves_pinned)
    # The curves of the paper's figure: same image-angle pairing as train.py --test-only
    # (jpeg_gap --order pinned), so they agree with the tables. One backbone per call:
    #   bash eval/trace_final_batch.sh quality_curves_pinned convnextv2_base   (or efficientvit_b3, mambaout_base)
    BB="${2:?backbone: convnextv2_base, efficientvit_b3 or mambaout_base}"
    CONDS="clean,jpeg95,jpeg90,jpeg80,jpeg70,jpeg60,jpeg50,webp90"
    OUTD="$AUD/quality_curve_pinned"; mkdir -p "$OUTD"
    declare -A SHORT=([convnextv2_base]=convnext [efficientvit_b3]=effvit [mambaout_base]=mambaout)
    for pair in "camera_${SHORT[$BB]}:$(ck_of "cgd_${BB}_coco_camera")" \
                "paper_${SHORT[$BB]}:$(grep "^cgd|${BB}|" eval/matrix_models_all.txt | cut -d'|' -f3)"; do
      name="${pair%%:*}"; ck="${pair#*:}"
      [[ -f "$OUTD/$name.json" ]] && continue
      echo "======== CURVE (pinned order) $name $(date '+%F %T') ========"
      python eval/jpeg_gap.py --approach cgd --ckpt "$ck" --seeds 0,1,2,3,4 --conditions "$CONDS" --order pinned \
        --out "$OUTD/$name.json" 2>&1 | grep -v "INFO\|Warning"
    done
    ;;
  residue)
    GROUP="${2:?group}"
    cat "$LISTS"/models_*"$GROUP"*.txt > "$LISTS/residue_$GROUP.txt"
    echo "RESIDUE $GROUP: $(wc -l < "$LISTS/residue_$GROUP.txt") models $(date '+%F %T')"
    python eval/rotation_residue.py --models "$LISTS/residue_$GROUP.txt" --out-dir "$AUD/rotation_residue"
    ;;
  figures)
    mkdir -p eval/results/figures_release
    # q90 is the primary test condition; 0 (no re-encode) for the published-protocol column
    for Q in 90 0; do
      python eval/generate_error_histogram.py --checkpoint "$(ck_of "$RELEASE_TAG")" --jpeg-quality $Q \
        --output eval/results/figures_release/error_histogram_q$Q.pdf
      python eval/generate_qualitative_figure.py --checkpoint "$(ck_of "$RELEASE_TAG")" --jpeg-quality $Q \
        --output-dir eval/results/figures_release/qualitative_q$Q
    done
    ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
echo "FINAL_BATCH $MODE ${2:-} DONE $(date '+%F %T')"
