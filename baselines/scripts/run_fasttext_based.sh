#!/bin/bash

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model>"
    echo "Available models: glotlid, openlid, openlid-v2, retrained"
    exit 1
fi

MODEL=$1
#LANG200=data/OpenLID-v2/languages.txt

if [ "$MODEL" = "openlid" ] || [ "$MODEL" = "openlid-v2" ]; then
    ARGS="--model $MODEL --enable-preprocessing"
elif [ "$MODEL" = "retrained" ]; then
    ARGS="--model retrained --model-path /scratch/project_465002259/OpenLID-v2/model.bin --enable-preprocessing"
else
    ARGS="--model $MODEL"
fi

mkdir -p results

echo "Running $MODEL on flores plus dev"
python3 scripts/fasttext_predictions.py --dataset flores --split dev $ARGS > results/flores_plus_dev_${MODEL}_predictions.jsonl
#python3 scripts/fasttext_predictions.py --dataset flores --split dev --languages-file $LANG200 $ARGS > results/flores_plus_dev_${MODEL}_lang200_predictions.jsonl

echo "Running $MODEL on flores plus devtest"
python3 scripts/fasttext_predictions.py --dataset flores --split devtest $ARGS > results/flores_plus_devtest_${MODEL}_predictions.jsonl
#python3 scripts/fasttext_predictions.py --dataset flores --split devtest --languages-file $LANG200 $ARGS > results/flores_plus_devtest_${MODEL}_lang200_predictions.jsonl

echo "Running $MODEL on udhr"
python3 scripts/fasttext_predictions.py --dataset udhr $ARGS > results/udhr_${MODEL}_predictions.jsonl
#python3 scripts/fasttext_predictions.py --dataset udhr --languages-file $LANG200 $ARGS > results/udhr_${MODEL}_lang200_predictions.jsonl

echo "Completed running $MODEL on both splits"
