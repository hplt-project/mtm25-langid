#!/bin/bash

set -e

# Check if model parameter is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model>"
    echo "Available models: glotlid, openlid"
    echo "Example: $0 glotlid"
    echo "Example: $0 openlid"
    exit 1
fi

MODEL=$1

# Validate model parameter
if [ "$MODEL" != "glotlid" ] && [ "$MODEL" != "openlid" ]; then
    echo "Error: Invalid model '$MODEL'"
    echo "Available models: glotlid, openlid"
    exit 1
fi

mkdir -p results

echo "Running $MODEL on dev"
python3 scripts/${MODEL}.py dev > results/flores_plus_dev_${MODEL}_predictions.jsonl

echo "Running $MODEL on devtest"
python3 scripts/${MODEL}.py devtest > results/flores_plus_devtest_${MODEL}_predictions.jsonl

echo "Completed running $MODEL on both splits"
