#!/bin/bash

set -e

# Add the ConLID submodule to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/3rd_party/ConLID"

mkdir -p results

echo "Running conlid on dev"
python3 scripts/conlid_predictions.py --dataset flores --model conlid --model-dir models/conlid-model --split dev > results/flores_plus_dev_conlid_predictions.jsonl

echo "Running conlid on devtest"
python3 scripts/conlid_predictions.py --dataset flores --model conlid --model-dir models/conlid-model --split devtest > results/flores_plus_devtest_conlid_predictions.jsonl

echo "Running conlid on udhr"
python3 scripts/conlid_predictions.py --dataset udhr --model conlid --model-dir models/conlid-model > results/udhr_conlid_predictions.jsonl

echo "Completed running conlid on both splits"
