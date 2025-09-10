#!/bin/bash

mkdir -p models/glot500_finetuned

python3 scripts/finetune_glot500.py \
    --train-data data/OpenLID-v2/train.jsonl \
    --eval-data data/flores_plus/dev.jsonl \
    --languages-file data/OpenLID-v2/languages.txt \
    --output-dir models/glot500_finetuned \
    --num-epochs 1 \
    --batch-size 32

echo "Fine-tuning completed! Model saved to models/glot500_finetuned/"
