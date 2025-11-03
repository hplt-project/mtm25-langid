#!/bin/bash
#SBATCH --account=project_465002259
#SBATCH --job-name=conlid
#SBATCH --output=logs/conlid_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

export EBU_USER_PREFIX=/projappl/project_465002310/software

module --quiet purge
module load LUMI
module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250404

export LC_ALL=en_us.UTF-8

if [ ! -d "models/conlid-model" ]; then
    echo "Downloading conlid model"
    srun singularity exec $SIF python3 scripts/download_conlid_model.py
fi

srun singularity exec $SIF bash scripts/run_conlid.sh
