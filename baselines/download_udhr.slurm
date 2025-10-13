#!/bin/bash
#SBATCH --account=project_465002259
#SBATCH --job-name=download_udhr
#SBATCH --output=logs/download_udhr_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

export EBU_USER_PREFIX=/projappl/project_465002310/software

module --quiet purge
module load LUMI
module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250404

export LC_ALL=en_us.UTF-8

srun singularity exec $SIF python3 scripts/download_udhr.py
