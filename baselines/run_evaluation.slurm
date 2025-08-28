#!/bin/bash
#SBATCH --account=project_465001925
#SBATCH --job-name=evaluate_all
#SBATCH --output=logs/evaluate_all_%j.out
#SBATCH --error=logs/evaluate_all_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

export EBU_USER_PREFIX=/projappl/project_465001925/software

module --quiet purge
module load LUMI
module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250404

export LC_ALL=en_us.UTF-8

mkdir -p results

python3 scripts/evaluate.py --model glotlid --dataset flores results/flores_plus_dev_glotlid_predictions.jsonl > results/flores_plus_dev_glotlid_evaluation.json
python3 scripts/evaluate.py --model glotlid --dataset flores results/flores_plus_devtest_glotlid_predictions.jsonl > results/flores_plus_devtest_glotlid_evaluation.json
python3 scripts/evaluate.py --model glotlid --dataset udhr results/udhr_glotlid_predictions.jsonl > results/udhr_glotlid_evaluation.json
python3 scripts/evaluate.py --model openlid --dataset flores results/flores_plus_dev_openlid_predictions.jsonl > results/flores_plus_dev_openlid_evaluation.json
python3 scripts/evaluate.py --model openlid --dataset flores results/flores_plus_devtest_openlid_predictions.jsonl > results/flores_plus_devtest_openlid_evaluation.json
python3 scripts/evaluate.py --model openlid --dataset udhr results/udhr_openlid_predictions.jsonl > results/udhr_openlid_evaluation.json

# Glot500 evaluations
#python3 scripts/evaluate.py --model glot500 --dataset flores results/flores_plus_dev_glot500_predictions.jsonl > results/flores_plus_dev_glot500_evaluation.json
#python3 scripts/evaluate.py --model glot500 --dataset flores results/flores_plus_devtest_glot500_predictions.jsonl > results/flores_plus_devtest_glot500_evaluation.json
#python3 scripts/evaluate.py --model glot500 --dataset udhr results/udhr_glot500_predictions.jsonl > results/udhr_glot500_evaluation.json
