#!/bin/bash

set -e

# Check if model directory parameter is provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_dir> <languages_file>"
    echo "Example: $0 models/glot500_finetuned_20M languages.txt"
    exit 1
fi

MODEL_BASE_DIR=$1
LANGUAGES_FILE=$2

# Find the checkpoint with the highest number
CHECKPOINT_DIR=$(ls -d "$MODEL_BASE_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Error: No checkpoint-* directories found in $MODEL_BASE_DIR"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT_DIR"

MODEL=$(basename "$MODEL_BASE_DIR")

mkdir -p results

echo "Running $MODEL on dev"
python3 scripts/glot500_predictions.py --dataset flores --model $MODEL --model-dir "$CHECKPOINT_DIR" --languages-file "$LANGUAGES_FILE" --split dev > results/flores_plus_dev_${MODEL}_predictions.jsonl

echo "Running $MODEL on devtest"
python3 scripts/glot500_predictions.py --dataset flores --model $MODEL --model-dir "$CHECKPOINT_DIR" --languages-file "$LANGUAGES_FILE" --split devtest > results/flores_plus_devtest_${MODEL}_predictions.jsonl

echo "Running $MODEL on udhr"
python3 scripts/glot500_predictions.py --dataset udhr --model $MODEL --model-dir "$CHECKPOINT_DIR" --languages-file "$LANGUAGES_FILE" > results/udhr_${MODEL}_predictions.jsonl

echo "Completed running $MODEL on all datasets"
