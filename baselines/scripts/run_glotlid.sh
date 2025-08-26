#!/bin/bash

set -e

mkdir -p results

echo "running on dev"
python3 scripts/glotlid.py dev > results/flores_plus_dev_glotlid_predictions.jsonl 2>logs/glotlid_dev.log

echo "running on devtest"
python3 scripts/glotlid.py devtest > results/flores_plus_devtest_glotlid_predictions.jsonl 2>logs/glotlid_devtest.log
