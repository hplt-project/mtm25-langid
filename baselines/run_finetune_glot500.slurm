#!/bin/bash
#SBATCH --account=project_465001925
#SBATCH --job-name=finetune_glot500
#SBATCH --output=logs/finetune_glot500_%j.out
#SBATCH --error=logs/finetune_glot500_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4

export EBU_USER_PREFIX=/projappl/project_465001925/software

module --quiet purge
module load LUMI
module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250404

export LC_ALL=en_us.UTF-8

mkdir -p models/glot500_finetuned

torchrun --nproc_per_node=4 --master_port=29500 scripts/finetune_glot500.py \
    --train_data data/openlid/train.jsonl \
    --eval_data baselines/data/flores_plus/dev.jsonl \
    --output_dir models/glot500_finetuned \
    --num_epochs 3 \
    --batch_size 16

echo "Fine-tuning completed! Model saved to models/glot500_finetuned/"
