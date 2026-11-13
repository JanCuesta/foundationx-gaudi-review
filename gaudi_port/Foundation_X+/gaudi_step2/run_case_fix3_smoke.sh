#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE="${1:?Usage: run_case.sh h|i|j|k|l smoke|train}"
MODE="${2:?Usage: run_case.sh h|i|j|k|l smoke|train}"
[[ "$CASE" =~ ^[hijkl]$ ]] || { echo "Invalid case" >&2; exit 2; }
[[ "$MODE" == smoke || "$MODE" == train ]] || exit 2
DATA=$(awk -v c="$CASE" '$1==c {print $0}' "$ROOT/gaudi_step2/cases.txt")
read -r NAME TASK_COUNT TASKS <<< "$DATA"
[[ "$NAME" == "$CASE" && -n "$TASKS" ]] || exit 2
EPOCHS_PER_TASK="${FX_EPOCHS_PER_TASK:-100}"
[[ "$EPOCHS_PER_TASK" =~ ^[0-9]+$ ]] || exit 2
if [[ "$MODE" == smoke ]]; then
  OUT="$ROOT/gaudi_step2/results/smoke_$CASE"
  TOTAL_EPOCHS=$((TASK_COUNT+1))
  EXTRA=(--debug)
  BATCH=2
else
  # Explicit gate: don't allow long jobs before manual review of HPU smoke logs.
  test -f "$ROOT/gaudi_step2/HPU_SMOKE_APPROVED" || { echo 'Smoke tests must pass before full training; missing HPU_SMOKE_APPROVED' >&2; exit 7; }
  OUT="$ROOT/gaudi_step2/results/fix3_smoke_$CASE"
  TOTAL_EPOCHS=$((TASK_COUNT+1))
  EXTRA=(--debug)
  BATCH=2
fi
mkdir -p "$OUT"
# Resume only from a nonempty checkpoint already matching this experiment.
RESUME=()
if [[ "$MODE" == train ]]; then
  LAST=$(find "$OUT" -maxdepth 1 -name 'ckpt_E*_TH*.pth' -type f -size +100M -printf '%f\n' | sort -V | tail -1 || true)
  if [[ -n "$LAST" ]]; then
    RESUME=(--resume "$OUT/$LAST")
    echo "RESUME: $OUT/$LAST"
  fi
fi
cd "$ROOT"
# Intel's Gaudi guidance recommends Lazy mode and disabling dynamic-shape
# refinement for this training path. Both remain explicitly overridable.
export PT_HPU_LAZY_MODE="${PT_HPU_LAZY_MODE:-1}"
export PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES="${PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES:-0}"
export PYTHONPATH="$ROOT:$ROOT/gaudi_step2/pydeps:$(dirname "$ROOT")/.gaudi_pydeps:${PYTHONPATH:-}"
python -u main_Consolidated.py --device hpu --taskcomponent foundation_x5_pretraining --train --numClasses 1 --dataset_file foundation6Ark6_datasets --classification_dataset foundation6Ark6_datasets --num_workers 4 --coco_path /data/jliang12/bmalamut/CANDID-PTX/png --weight-decay 0.0001 --output_dir "$OUT" -c config/DINO/DINO_4scale_swinBASE.py --imgsize 224 --backbonemodel Swin-B --init ark --total_epochs "$TOTAL_EPOCHS" --batch_size "$BATCH" --opt adamw --finetune_ignore label_enc.weight class_embed --backbone_dir /data/jliang12/dongaoma/Ark_models/TSconsist_NoOD_MIMIC_CheXpert_ChestXray14_RSNAPneumonia_VinDrCXR_Shenzhen_ep200.pth.tar --lr_backbone 1e-5 --lr_locEnc 1e-4 --lr_locDec 1e-4 --lr_segmentor 1e-4 --cyclictask "$TASKS" --serverC SOL --modelEMA True_Epoch --lockrelease --saveAllModel "${RESUME[@]}" "${EXTRA[@]}" --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0
