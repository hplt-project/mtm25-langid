#!/usr/bin/env python3

import os
import json
from pathlib import Path
from datasets import load_dataset


def download_udhr():
    output_dir = Path("data/udhr")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading UDHR-LID dataset...")
    dataset = load_dataset("cis-lmu/udhr-lid", split="test")

    print(f"Processing test split with {len(dataset)} examples...")

    processed_data = []
    for example in dataset:
        processed_data.append({
            "text": example["sentence"],
            "language": example["iso639-3"],
            "script": example["iso15924"],
            "id": example["id"]
        })

    output_file = output_dir / "test.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(processed_data)} examples to {output_file}")


if __name__ == "__main__":
    download_udhr()
