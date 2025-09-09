#!/bin/bash
#SBATCH --job-name=preprocess_openlid
#SBATCH --output=logs/preprocess_openlid_%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=cpu-troja,cpu-ms
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G

source env/bin/activate
time python3 scripts/preprocess_openlid.py \
       --input data/OpenLID-v2/train.jsonl \
       --output data/OpenLID-v2/preprocessed_train \
       --languages-file data/OpenLID-v2/languages.txt
