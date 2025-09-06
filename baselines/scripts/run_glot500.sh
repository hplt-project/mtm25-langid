#!/bin/bash

set -e

# Check if model directory parameter is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_dir>"
    echo "Example: $0 models/glot500_finetuned_20M/checkpoint-382000"
    exit 1
fi

MODEL_DIR=$1
MODEL=$(basename "$MODEL_DIR")

mkdir -p results

echo "Running $MODEL on dev"
python3 scripts/glot500_predictions.py --dataset flores --model $MODEL --model-dir "$MODEL_DIR" --split dev > results/flores_plus_dev_${MODEL}_predictions.jsonl

echo "Running $MODEL on devtest"
python3 scripts/glot500_predictions.py --dataset flores --model $MODEL --model-dir "$MODEL_DIR" --split devtest > results/flores_plus_devtest_${MODEL}_predictions.jsonl

echo "Running $MODEL on udhr"
python3 scripts/glot500_predictions.py --dataset udhr --model $MODEL --model-dir "$MODEL_DIR" > results/udhr_${MODEL}_predictions.jsonl

echo "Completed running $MODEL on all datasets"
