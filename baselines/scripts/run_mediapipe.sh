#!/bin/bash

set -e

mkdir -p predictions

echo "Running mediapipe on dev"
time python3 scripts/mediapipe_predictions.py --dataset flores --split dev > predictions/mediapipe.flores_plus_dev.txt

echo "Running mediapipe on devtest"
time python3 scripts/mediapipe_predictions.py --dataset flores --split devtest > predictions/mediapipe.flores_plus_devtest.txt

echo "Running mediapipe on udhr"
time python3 scripts/mediapipe_predictions.py --dataset udhr > predictions/mediapipe.udhr.txt
