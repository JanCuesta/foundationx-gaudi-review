#!/bin/bash
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH --gres=gpu:hl225:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -J p2_j_roi
#SBATCH -o /scratch/jcuesta/BME/Foundation_X/gaudi_port/Foundation_X+/gaudi_step2/diagnostics/seg_j_roi/logs/seg_j_roi_%j.out
#SBATCH -e /scratch/jcuesta/BME/Foundation_X/gaudi_port/Foundation_X+/gaudi_step2/diagnostics/seg_j_roi/logs/seg_j_roi_%j.err
set -euo pipefail
/scratch/jcuesta/BME/Practice3_UniMiSSPlus/Gaudi_Demo/gaudi-apptainer.sh exec bash /scratch/jcuesta/BME/Foundation_X/gaudi_port/Foundation_X+/gaudi_step2/diagnostics/seg_j_roi/run_seg_roi_probe.sh j train
