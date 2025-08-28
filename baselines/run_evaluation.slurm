#!/bin/bash
#SBATCH --job-name=evaluate_all
#SBATCH --output=logs/evaluate_all_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=cpu-troja,cpu-ms
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

source env/bin/activate

mkdir -p results

python3 scripts/evaluate.py --model glotlid --dataset flores results/flores_plus_dev_glotlid_predictions.jsonl --languages-file data/OpenLID-v2/languages.txt > results/flores_plus_dev_glotlid_evaluation.json
python3 scripts/evaluate.py --model glotlid --dataset flores results/flores_plus_devtest_glotlid_predictions.jsonl --languages-file data/OpenLID-v2/languages.txt > results/flores_plus_devtest_glotlid_evaluation.json
python3 scripts/evaluate.py --model glotlid --dataset udhr results/udhr_glotlid_predictions.jsonl --languages-file data/OpenLID-v2/languages.txt > results/udhr_glotlid_evaluation.json

python3 scripts/evaluate.py --model glotlid --dataset flores results/flores_plus_dev_glotlid_lang200_predictions.jsonl --languages-file data/OpenLID-v2/languages.txt > results/flores_plus_dev_glotlid_lang200_evaluation.json
python3 scripts/evaluate.py --model glotlid --dataset flores results/flores_plus_devtest_glotlid_lang200_predictions.jsonl --languages-file data/OpenLID-v2/languages.txt > results/flores_plus_devtest_glotlid_lang200_evaluation.json
python3 scripts/evaluate.py --model glotlid --dataset udhr results/udhr_glotlid_lang200_predictions.jsonl --languages-file data/OpenLID-v2/languages.txt > results/udhr_glotlid_lang200_evaluation.json

python3 scripts/evaluate.py --model openlid-v2 --dataset flores results/flores_plus_dev_openlid-v2_predictions.jsonl --languages-file data/OpenLID-v2/languages.txt > results/flores_plus_dev_openlid-v2_evaluation.json
python3 scripts/evaluate.py --model openlid-v2 --dataset flores results/flores_plus_devtest_openlid-v2_predictions.jsonl --languages-file data/OpenLID-v2/languages.txt > results/flores_plus_devtest_openlid-v2_evaluation.json
python3 scripts/evaluate.py --model openlid-v2 --dataset udhr results/udhr_openlid-v2_predictions.jsonl --languages-file data/OpenLID-v2/languages.txt > results/udhr_openlid-v2_evaluation.json

#python3 scripts/evaluate.py --model glot500_finetuned_20M --dataset flores --languages-file data/OpenLID-v2/languages.txt results/flores_plus_dev_glot500_finetuned_20M_predictions.jsonl > results/flores_plus_dev_glot500_finetuned_20M_evaluation.json
#python3 scripts/evaluate.py --model glot500_finetuned_20M --dataset flores --languages-file data/OpenLID-v2/languages.txt results/flores_plus_devtest_glot500_finetuned_20M_predictions.jsonl > results/flores_plus_devtest_glot500_finetuned_20M_evaluation.json
#python3 scripts/evaluate.py --model glot500_finetuned_20M --dataset udhr --languages-file data/OpenLID-v2/languages.txt results/udhr_glot500_finetuned_20M_predictions.jsonl > results/udhr_glot500_finetuned_20M_evaluation.json

python3 scripts/evaluate.py --model conlid --dataset flores --languages-file data/OpenLID-v2/languages.txt results/flores_plus_dev_conlid_predictions.jsonl > results/flores_plus_dev_conlid_evaluation.json
python3 scripts/evaluate.py --model conlid --dataset flores --languages-file data/OpenLID-v2/languages.txt results/flores_plus_devtest_conlid_predictions.jsonl > results/flores_plus_devtest_conlid_evaluation.json
python3 scripts/evaluate.py --model conlid --dataset udhr --languages-file data/OpenLID-v2/languages.txt results/udhr_conlid_predictions.jsonl > results/udhr_conlid_evaluation.json
