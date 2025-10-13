#!/bin/bash

mkdir -p models/glot500_finetuned

python3 scripts/finetune_glot500.py \
    --train-path data/OpenLID-v2/preprocessed_train \
    --eval-data data/flores_plus/dev.jsonl \
    --languages-file data/OpenLID-v2/languages.txt \
    --output-dir models/glot500_finetuned_unfreezing \
    --gradual-unfreezing \
    --num-epochs 1 \
    --batch-size 32

echo "Fine-tuning completed! Model saved to models/glot500_finetuned/"
