#!/bin/bash

set -e

python3 scripts/gather_predictions.py --dataset flores --split dev \
    --prediction conlid predictions/conlid.flores_plus_dev.txt \
    --prediction openlid-v2 predictions/openlid-v2.flores_plus_dev.txt \
    --prediction retrained predictions/retrained.flores_plus_dev.txt \
    --prediction glotlid predictions/glotlid.flores_plus_dev.txt \
    --output predictions/flores_plus_dev.jsonl

python3 scripts/gather_predictions.py --dataset flores --split devtest \
    --prediction conlid predictions/conlid.flores_plus_devtest.txt \
    --prediction openlid-v2 predictions/openlid-v2.flores_plus_devtest.txt \
    --prediction retrained predictions/retrained.flores_plus_devtest.txt \
    --prediction glotlid predictions/glotlid.flores_plus_devtest.txt \
    --output predictions/flores_plus_devtest.jsonl

python3 scripts/gather_predictions.py --dataset udhr \
    --prediction conlid predictions/conlid.udhr.txt \
    --prediction openlid-v2 predictions/openlid-v2.udhr.txt \
    --prediction retrained predictions/retrained.udhr.txt \
    --prediction glotlid predictions/glotlid.udhr.txt \
    --output predictions/udhr.jsonl
