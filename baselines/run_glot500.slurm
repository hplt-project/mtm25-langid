#!/bin/bash
#SBATCH --job-name=glot500
#SBATCH --output=logs/glot500_%j.out
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-troja,gpu-ms
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source env/bin/activate
time ./scripts/run_glot500.sh models/glot500_finetuned_20M/checkpoint-210000
