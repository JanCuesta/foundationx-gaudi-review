#!/bin/bash
#SBATCH --job-name=p2_100_e_cls_seg
#SBATCH -N 1
#SBATCH -G a100:1
#SBATCH -c 12
#SBATCH --mem=80G
#SBATCH -p public
#SBATCH -q class
#SBATCH -t 07:55:00
#SBATCH -o /scratch/jcuesta/BME/Foundation_X/p2_100_e_cls_seg_%j.out
#SBATCH -e /scratch/jcuesta/BME/Foundation_X/p2_100_e_cls_seg_%j.err

module load cuda-11.6.2-gcc-12.1.0
ENV=/home/jcuesta/.conda/envs/foundationx
PY=$ENV/bin/python
export PATH=$ENV/bin:$PATH
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export LD_LIBRARY_PATH=$ENV/lib:$ENV/lib/python3.7/site-packages/torch/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
cd /scratch/jcuesta/BME/Foundation_X/Foundation_X+
if ! $PY -c "import MultiScaleDeformableAttention" >/dev/null 2>&1; then cd models/dino/ops && rm -rf build && TORCH_CUDA_ARCH_LIST=8.0 $PY setup.py build install && cd ../../..; fi
$PY -c "import torch,MultiScaleDeformableAttention; print(\"GPU:\",torch.cuda.get_device_name(0)); print(\"MSDA OK\")"
OUT=/scratch/jcuesta/BME/Foundation_X/results/p2_100_e_cls_seg
mkdir -p "$OUT"
# AUTO_RESUME
LAST_EPOCH=""
LATEST=""
if [ -f "$OUT/export_csvFile.csv" ]; then LAST_EPOCH=$(tail -n +2 "$OUT/export_csvFile.csv" | cut -d, -f1 | grep -E "^[0-9]+$" | sort -n | tail -1); fi
if [ -n "$LAST_EPOCH" ]; then LATEST=$(find "$OUT" -maxdepth 1 -type f -name "ckpt_E${LAST_EPOCH}_TH*.pth" | head -1); fi
if [ -z "$LATEST" ]; then LATEST=$(find "$OUT" -maxdepth 1 -type f -name "ckpt_E*_TH*.pth" | sort -V | tail -1); fi
RESUME_ARG=""
if [ -n "$LATEST" ]; then RESUME_ARG="--resume $LATEST"; echo "Resuming from $LATEST"; else echo "Starting from scratch"; fi

P2_STOP_AFTER_EPOCH=85 $PY main_Consolidated_p2_limited.py --taskcomponent foundation_x5_pretraining --train --numClasses 1 --dataset_file foundation6Ark6_datasets --classification_dataset foundation6Ark6_datasets --num_workers 12 --coco_path /scratch/jcuesta/BME/datasets/CANDID-PTX/png --weight-decay 0.0001 --output_dir $OUT -c config/DINO/DINO_4scale_swinBASE.py --imgsize 224 --backbonemodel Swin-B --init ark --total_epochs 201 --batch_size 24 --opt adamw --finetune_ignore label_enc.weight class_embed --backbone_dir /scratch/sejong/class-dataset/models/Ark_models/TSconsist_NoOD_MIMIC_CheXpert_ChestXray14_RSNAPneumonia_VinDrCXR_Shenzhen_ep200.pth.tar --lr_backbone 1e-5 --lr_locEnc 1e-4 --lr_locDec 1e-4 --lr_segmentor 1e-4 --cyclictask candidptxCLS_candidptxSEG --serverC SOL --modelEMA True_Epoch --lockrelease --saveAllModel $RESUME_ARG --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0
