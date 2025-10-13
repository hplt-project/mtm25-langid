#!/bin/bash
#SBATCH --account=project_465002259
#SBATCH --job-name=finetune_glot500
#SBATCH --output=logs/finetune_glot500_%j.out
#SBATCH --time=72:00:00
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4

export EBU_USER_PREFIX=/projappl/project_465002310/software

module --quiet purge
module load LUMI
module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250404

export LC_ALL=en_us.UTF-8

srun singularity exec $SIF bash scripts/finetune_glot500.sh
