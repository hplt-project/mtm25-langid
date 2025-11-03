#!/bin/bash
#SBATCH --account=project_465002259
#SBATCH --job-name=evaluate_all
#SBATCH --output=logs/evaluate_all_%j.out
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

#LANG_LIST_FILE="data/OpenLID-v2/languages.txt"
LANG_LIST_FILE="language-lists/openlid-flores-glotlid-udhr.txt"

mkdir -p results

python3 scripts/evaluate.py --model conlid --dataset flores predictions/flores_plus_dev.jsonl --languages-file $LANG_LIST_FILE > results/flores_plus_dev_conlid_evaluation.json
python3 scripts/evaluate.py --model conlid --dataset flores predictions/flores_plus_devtest.jsonl --languages-file $LANG_LIST_FILE > results/flores_plus_devtest_conlid_evaluation.json
python3 scripts/evaluate.py --model conlid --dataset udhr predictions/udhr.jsonl --languages-file $LANG_LIST_FILE > results/udhr_conlid_evaluation.json

python3 scripts/evaluate.py --model glotlid --dataset flores predictions/flores_plus_dev.jsonl --languages-file $LANG_LIST_FILE > results/flores_plus_dev_glotlid_evaluation.json
python3 scripts/evaluate.py --model glotlid --dataset flores predictions/flores_plus_devtest.jsonl --languages-file $LANG_LIST_FILE > results/flores_plus_devtest_glotlid_evaluation.json
python3 scripts/evaluate.py --model glotlid --dataset udhr predictions/udhr.jsonl --languages-file $LANG_LIST_FILE > results/udhr_glotlid_evaluation.json

python3 scripts/evaluate.py --model openlid-v2 --dataset flores predictions/flores_plus_dev.jsonl --languages-file $LANG_LIST_FILE > results/flores_plus_dev_openlid-v2_evaluation.json
python3 scripts/evaluate.py --model openlid-v2 --dataset flores predictions/flores_plus_devtest.jsonl --languages-file $LANG_LIST_FILE > results/flores_plus_devtest_openlid-v2_evaluation.json
python3 scripts/evaluate.py --model openlid-v2 --dataset udhr predictions/udhr.jsonl --languages-file $LANG_LIST_FILE > results/udhr_openlid-v2_evaluation.json

python3 scripts/evaluate.py --model retrained --dataset flores predictions/flores_plus_dev.jsonl --languages-file $LANG_LIST_FILE > results/flores_plus_dev_retrained_evaluation.json
python3 scripts/evaluate.py --model retrained --dataset flores predictions/flores_plus_devtest.jsonl --languages-file $LANG_LIST_FILE > results/flores_plus_devtest_retrained_evaluation.json
python3 scripts/evaluate.py --model retrained --dataset udhr predictions/udhr.jsonl --languages-file $LANG_LIST_FILE > results/udhr_retrained_evaluation.json
