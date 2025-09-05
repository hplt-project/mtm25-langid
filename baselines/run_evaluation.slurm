#!/bin/bash
#SBATCH --job-name=evaluate_all
#SBATCH --output=logs/evaluate_all_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=cpu-troja,cpu-ms
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

source env/bin/activate

mkdir -p results

# flores dev
python3 scripts/evaluate.py --model glotlid --dataset flores --languages-file data/openlid/languages.txt results/flores_plus_dev_glotlid_predictions.jsonl > results/flores_plus_dev_glotlid_evaluation.json
python3 scripts/evaluate.py --model openlid --dataset flores --languages-file data/openlid/languages.txt results/flores_plus_dev_openlid_predictions.jsonl > results/flores_plus_dev_openlid_evaluation.json
python3 scripts/evaluate.py --model conlid --dataset flores --languages-file data/openlid/languages.txt results/flores_plus_dev_conlid_predictions.jsonl > results/flores_plus_dev_conlid_evaluation.json
#python3 scripts/evaluate.py --model glot500 --dataset flores --languages-file data/openlid/languages.txt results/flores_plus_dev_glot500_predictions.jsonl > results/flores_plus_dev_glot500_evaluation.json

# flores devtest
python3 scripts/evaluate.py --model glotlid --dataset flores --languages-file data/openlid/languages.txt results/flores_plus_devtest_glotlid_predictions.jsonl > results/flores_plus_devtest_glotlid_evaluation.json
python3 scripts/evaluate.py --model openlid --dataset flores --languages-file data/openlid/languages.txt results/flores_plus_devtest_openlid_predictions.jsonl > results/flores_plus_devtest_openlid_evaluation.json
python3 scripts/evaluate.py --model conlid --dataset flores --languages-file data/openlid/languages.txt results/flores_plus_devtest_conlid_predictions.jsonl > results/flores_plus_devtest_conlid_evaluation.json
#python3 scripts/evaluate.py --model glot500 --dataset flores --languages-file data/openlid/languages.txt results/flores_plus_devtest_glot500_predictions.jsonl > results/flores_plus_devtest_glot500_evaluation.json

# udhr
python3 scripts/evaluate.py --model glotlid --dataset udhr --languages-file data/openlid/languages.txt results/udhr_glotlid_predictions.jsonl > results/udhr_glotlid_evaluation.json
python3 scripts/evaluate.py --model openlid --dataset udhr --languages-file data/openlid/languages.txt results/udhr_openlid_predictions.jsonl > results/udhr_openlid_evaluation.json
python3 scripts/evaluate.py --model conlid --dataset udhr --languages-file data/openlid/languages.txt results/udhr_conlid_predictions.jsonl > results/udhr_conlid_evaluation.json
#python3 scripts/evaluate.py --model glot500 --dataset udhr --languages-file data/openlid/languages.txt results/udhr_glot500_predictions.jsonl > results/udhr_glot500_evaluation.json
