#!/bin/bash

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_base_dir>"
    exit 1
fi

MODEL_BASE_DIR=$1

# Find the checkpoint with the highest number
CHECKPOINT_DIR=$(ls -d "$MODEL_BASE_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Error: No checkpoint-* directories found in $MODEL_BASE_DIR"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT_DIR"

MODEL=$(basename "$MODEL_BASE_DIR")

mkdir -p predictions

echo "Running $MODEL on dev"
time python3 scripts/glot500_predictions.py --dataset flores --split dev --model-dir "$CHECKPOINT_DIR" > predictions/${MODEL}.flores_plus_dev.txt

echo "Running $MODEL on devtest"
time python3 scripts/glot500_predictions.py --dataset flores --split devtest --model-dir "$CHECKPOINT_DIR" > predictions/${MODEL}.flores_plus_devtest.txt

echo "Running $MODEL on udhr"
time python3 scripts/glot500_predictions.py --dataset udhr --model-dir "$CHECKPOINT_DIR" > predictions/${MODEL}.udhr.txt
