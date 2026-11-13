#!/bin/bash
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH --gres=gpu:hl225:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -J p2_hpu_Btest
#SBATCH -o /scratch/jcuesta/BME/Foundation_X/gaudi_port/smoke_b/slurm_%j.out
set -e
/scratch/jcuesta/BME/Practice3_UniMiSSPlus/Gaudi_Demo/gaudi-apptainer.sh exec bash -lc 'export PYTHONPATH=/scratch/jcuesta/BME/Foundation_X/gaudi_port/.gaudi_pydeps:${PYTHONPATH:-}; export PT_HPU_LAZY_MODE=0; cd /scratch/jcuesta/BME/Foundation_X/gaudi_port/Foundation_X+; python -u main_Consolidated_p2_smoke.py --taskcomponent foundation_x5_pretraining --train --numClasses 1 --dataset_file foundation6Ark6_datasets --classification_dataset foundation6Ark6_datasets --num_workers 2 --coco_path /scratch/jcuesta/BME/datasets/CANDID-PTX/png --weight-decay 0.0001 --output_dir /scratch/jcuesta/BME/Foundation_X/gaudi_port/smoke_b -c config/DINO/DINO_4scale_swinBASE.py --imgsize 224 --backbonemodel Swin-B --init ark --total_epochs 101 --batch_size 2 --opt adamw --finetune_ignore label_enc.weight class_embed --backbone_dir /scratch/sejong/class-dataset/models/Ark_models/TSconsist_NoOD_MIMIC_CheXpert_ChestXray14_RSNAPneumonia_VinDrCXR_Shenzhen_ep200.pth.tar --lr_backbone 1e-5 --lr_locEnc 1e-4 --lr_locDec 1e-4 --lr_segmentor 1e-4 --cyclictask candidptxLOC --serverC SOL --modelEMA True_Epoch --lockrelease --saveAllModel --resume /scratch/jcuesta/BME/Foundation_X/results/p2_100_b_loc/ckpt_E43_TH10.pth --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 --device hpu > /scratch/jcuesta/BME/Foundation_X/gaudi_port/smoke_b/run.log 2>&1'
