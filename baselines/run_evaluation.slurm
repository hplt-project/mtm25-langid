#!/bin/bash
#SBATCH --job-name=evaluate_all
#SBATCH --output=logs/evaluate_all_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=cpu-troja,cpu-ms
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

source env/bin/activate

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
