#!/bin/bash
#SBATCH --account=project_465001925
#SBATCH --job-name=mediapipe
#SBATCH --output=logs/mediapipe_%j.out
#SBATCH --error=logs/mediapipe_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

export EBU_USER_PREFIX=/projappl/project_465001925/software

module --quiet purge
module load LUMI
module load PyTorch/2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617

export LC_ALL=en_us.UTF-8

srun singularity exec $SIF bash scripts/run_mediapipe.sh mediapipe
