#!/bin/bash
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH --gres=gpu:hl225:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -J p2_hpu_ATfix
#SBATCH -o /scratch/jcuesta/BME/Foundation_X/gaudi_port/diag_attn_fixed/slurm_%j.out
set -e
/scratch/jcuesta/BME/Practice3_UniMiSSPlus/Gaudi_Demo/gaudi-apptainer.sh exec bash -lc 'set -e; export PYTHONPATH=/scratch/jcuesta/BME/Foundation_X/gaudi_port/Foundation_X+:/scratch/jcuesta/BME/Foundation_X/gaudi_port/.gaudi_pydeps:${PYTHONPATH:-}; export PT_HPU_LAZY_MODE=0; cd /scratch/jcuesta/BME/Foundation_X/gaudi_port/Foundation_X+; python -u /scratch/jcuesta/BME/Foundation_X/gaudi_port/diag_attn_fixed/check.py > /scratch/jcuesta/BME/Foundation_X/gaudi_port/diag_attn_fixed/run.log 2>&1'
