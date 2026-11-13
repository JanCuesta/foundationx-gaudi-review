#!/bin/bash
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH --gres=gpu:hl225:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J p2_j_fix3smoke
#SBATCH -o /scratch/jcuesta/BME/Foundation_X/gaudi_port/Foundation_X+/gaudi_step2/jobs/p2_j_fix3smoke_%j.out
#SBATCH -e /scratch/jcuesta/BME/Foundation_X/gaudi_port/Foundation_X+/gaudi_step2/jobs/p2_j_fix3smoke_%j.err
set -euo pipefail
/scratch/jcuesta/BME/Practice3_UniMiSSPlus/Gaudi_Demo/gaudi-apptainer.sh exec bash /scratch/jcuesta/BME/Foundation_X/gaudi_port/Foundation_X+/gaudi_step2/run_case_fix3_smoke.sh j train
