#!/bin/bash
#SBATCH -A grp_jliang12
#SBATCH -p public
#SBATCH -q public
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=sg021
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 14:00:00
#SBATCH -J fx_long97
#SBATCH -o fx_long97_%j.out
set -e
module load mamba/latest
source activate foundationx
module load cuda-11.3.1-gcc-12.1.0
export LD_PRELOAD=/home/jcuesta/.conda/envs/foundationx/lib/libstdc++.so.6
cd /scratch/jcuesta/BME/Foundation_X/Foundation_X+
mkdir -p full_training/job97
export FX_STOP_AFTER_EPOCH=97
/home/jcuesta/.conda/envs/foundationx/bin/python -u main_Consolidated_stepIII_long.py --taskcomponent foundation_x5_pretraining --train --numClasses 1 --dataset_file foundation6Ark6_datasets --classification_dataset foundation6Ark6_datasets --num_workers 2 --coco_path /scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/ --output_dir smoke_test_cls3 -c config/DINO/DINO_4scale_swinBASE.py --imgsize 224 --backbonemodel Swin-B --init ark --total_epochs 2000 --batch_size 24 --opt adamw --weight-decay 0.0001 --backbone_dir /data/jliang12/dongaoma/Ark_models/TSconsist_NoOD_MIMIC_CheXpert_ChestXray14_RSNAPneumonia_VinDrCXR_Shenzhen_ep200.pth.tar --lr_backbone 1e-5 --lr_locEnc 1e-4 --lr_locDec 1e-4 --lr_segmentor 1e-4 --cyclictask candidptxCLS --serverC SOL --modelEMA True_Epoch --lockrelease --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 --cyclictask chexpertCLS_nihchestxray14CLS_vindrcxrCLS_nihshenzenCLS_mimic2CLS_tbx11kCLS_node21CLS_candidptxCLS_rsnapneumoniaCLS_chestxdetCLS_siimacrCLS_tbx11kLOC_node21LOC_candidptxLOC_rsnapneumoniaLOC_chestxdetLOC_siimacrLOC_candidptxSEG_chestxdetSEG_siimacrSEG --output_dir full_training/job97 --total_epochs 2000 --batch_size 24  --saveAllModel --resume full_training/job96/ckpt_E96_TH15.pth > full_training/job97/run.log 2>&1

grep -Fq "[FX] Clean stop after epoch 97" full_training/job97/run.log && test -s full_training/job97/ckpt_E97_TH16.pth
