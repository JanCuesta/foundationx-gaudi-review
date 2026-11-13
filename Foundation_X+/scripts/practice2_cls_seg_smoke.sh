#!/bin/bash
#SBATCH --job-name=p2_cls_seg_smoke
#SBATCH -N 1
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -p public
#SBATCH -q class
#SBATCH -t 00:45:00
#SBATCH -o /scratch/jcuesta/BME/Foundation_X/p2_cls_seg_smoke_%j.out
#SBATCH -e /scratch/jcuesta/BME/Foundation_X/p2_cls_seg_smoke_%j.err

module load cuda-11.6.2-gcc-12.1.0
ENV=/home/jcuesta/.conda/envs/foundationx
PY=$ENV/bin/python
export PATH=$ENV/bin:$PATH
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export LD_LIBRARY_PATH=$ENV/lib:$ENV/lib/python3.7/site-packages/torch/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
cd /scratch/jcuesta/BME/Foundation_X/Foundation_X+
if ! $PY -c "import MultiScaleDeformableAttention" >/dev/null 2>&1; then cd models/dino/ops && rm -rf build && TORCH_CUDA_ARCH_LIST=8.0 $PY setup.py build install && cd ../../..; fi
$PY -c "import torch,MultiScaleDeformableAttention; print(\"GPU:\",torch.cuda.get_device_name(0)); print(\"MSDA OK\")"
OUT=/scratch/jcuesta/BME/Foundation_X/results/p2_cls_seg_smoke
rm -rf $OUT && mkdir -p $OUT
$PY main_Consolidated.py --taskcomponent foundation_x5_pretraining --train --numClasses 1 --dataset_file foundation6Ark6_datasets --classification_dataset foundation6Ark6_datasets --num_workers 4 --coco_path /scratch/jcuesta/BME/datasets/CANDID-PTX/png --weight-decay 0.0001 --output_dir $OUT -c config/DINO/DINO_4scale_swinBASE.py --imgsize 224 --backbonemodel Swin-B --init ark --total_epochs 3 --batch_size 4 --opt adamw --finetune_ignore label_enc.weight class_embed --backbone_dir /scratch/sejong/class-dataset/models/Ark_models/TSconsist_NoOD_MIMIC_CheXpert_ChestXray14_RSNAPneumonia_VinDrCXR_Shenzhen_ep200.pth.tar --lr_backbone 1e-5 --lr_locEnc 1e-4 --lr_locDec 1e-4 --lr_segmentor 1e-4 --cyclictask candidptxCLS_candidptxSEG --serverC SOL --modelEMA True_Epoch --lockrelease --saveAllModel --debug --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0
