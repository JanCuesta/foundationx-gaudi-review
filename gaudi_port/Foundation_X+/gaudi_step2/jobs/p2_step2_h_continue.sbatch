#!/bin/bash
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH --gres=gpu:hl225:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 7-00:00:00
#SBATCH -J p2_h_continue
#SBATCH -o /scratch/jcuesta/BME/Foundation_X/gaudi_port/Foundation_X+/gaudi_step2/jobs/p2_h_continue_%j.out
#SBATCH -e /scratch/jcuesta/BME/Foundation_X/gaudi_port/Foundation_X+/gaudi_step2/jobs/p2_h_continue_%j.err
set -euo pipefail
if [[ -n "${FX_H_PREV:-}" ]]; then
  ST=$(sacct -j "$FX_H_PREV" -X -n -P --format=State | head -1 | cut -d "|" -f1)
  case "$ST" in TIMEOUT*|COMPLETED*) ;; *) echo "STOP: previous H job $FX_H_PREV ended $ST"; exit 1;; esac
  [[ -n "$(find "/scratch/jcuesta/BME/Foundation_X/gaudi_port/Foundation_X+/gaudi_step2/results/train_h" -maxdepth 1 -type f -name "ckpt_E*_TH*.pth" -size +100M -print -quit 2>/dev/null)" ]] || { echo "STOP: no saved H checkpoint"; exit 1; }
fi
if [[ -s "/scratch/jcuesta/BME/Foundation_X/gaudi_port/Foundation_X+/gaudi_step2/results/train_h/ckpt_E400_TH17.pth" ]]; then echo "H already reached epoch 400"; exit 0; fi
/scratch/jcuesta/BME/Practice3_UniMiSSPlus/Gaudi_Demo/gaudi-apptainer.sh exec bash /scratch/jcuesta/BME/Foundation_X/gaudi_port/Foundation_X+/gaudi_step2/run_case.sh h train
