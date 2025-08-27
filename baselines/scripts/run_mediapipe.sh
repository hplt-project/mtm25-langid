#!/bin/bash

set -e

mkdir -p results

echo "Running mediapipe on dev"
python3 scripts/mediapipe_predictions.py --dataset flores --split dev > results/flores_plus_dev_mediapipe_predictions.jsonl

echo "Running mediapipe on devtest"
python3 scripts/mediapipe_predictions.py --dataset flores --split devtest > results/flores_plus_devtest_mediapipe_predictions.jsonl

echo "Running mediapipe on udhr"
python3 scripts/mediapipe_predictions.py --dataset udhr > results/udhr_mediapipe_predictions.jsonl

echo "Completed running mediapipe on both splits"
