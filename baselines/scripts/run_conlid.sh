#!/bin/bash

set -e

# Add the ConLID submodule to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/3rd_party/ConLID"

mkdir -p predictions

echo "Running conlid on dev"
time python3 scripts/conlid_predictions.py --dataset flores --split dev --model-dir models/conlid-model > predictions/conlid.flores_plus_dev.txt

echo "Running conlid on devtest"
time python3 scripts/conlid_predictions.py --dataset flores --split devtest --model-dir models/conlid-model > predictions/conlid.flores_plus_devtest.txt

echo "Running conlid on udhr"
time python3 scripts/conlid_predictions.py --dataset udhr --model-dir models/conlid-model > predictions/conlid.udhr.txt
