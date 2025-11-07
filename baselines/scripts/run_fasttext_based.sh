#!/bin/bash

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model>"
    echo "Available models: glotlid, openlid, openlid-v2, retrained, retrained-repl"
    exit 1
fi

MODEL=$1

if [ "$MODEL" = "openlid" ] || [ "$MODEL" = "openlid-v2" ]; then
    ARGS="--model $MODEL --enable-preprocessing"
elif [ "$MODEL" = "retrained" ]; then
    ARGS="--model retrained --model-path /scratch/project_465002259/OpenLID-v2/model.bin --enable-preprocessing"
elif [ "$MODEL" = "retrained-repl" ]; then
    ARGS="--model retrained --model-path /scratch/project_465002259/OpenLID-v2/rep-model.bin --enable-preprocessing"
elif [ "$MODEL" = "specific-sca" ] ; then
    ARGS="--model retrained --model-path /scratch/project_465002259/eurolid/cascade-data/sca/model.bin --enable-preprocessing"
elif [ "$MODEL" = "specific-hsb" ] ; then
    ARGS="--model retrained --model-path /scratch/project_465002259/eurolid/cascade-data/hsb/model.bin --enable-preprocessing"
elif [ "$MODEL" = "specific-fas" ] ; then
    ARGS="--model retrained --model-path /scratch/project_465002259/eurolid/cascade-data/fas/model.bin --enable-preprocessing"
elif [ "$MODEL" = "specific-ara" ] ; then
    ARGS="--model retrained --model-path /scratch/project_465002259/eurolid/cascade-data/ara/model.bin --enable-preprocessing"
else
    ARGS="--model $MODEL"
fi

mkdir -p predictions

echo "Running $MODEL on flores plus dev"
time python3 scripts/fasttext_predictions.py --dataset flores --split dev $ARGS > predictions/${MODEL}.flores_plus_dev.txt

echo "Running $MODEL on flores plus devtest"
time python3 scripts/fasttext_predictions.py --dataset flores --split devtest $ARGS > predictions/${MODEL}.flores_plus_devtest.txt

echo "Running $MODEL on udhr"
time python3 scripts/fasttext_predictions.py --dataset udhr $ARGS > predictions/${MODEL}.udhr.txt
