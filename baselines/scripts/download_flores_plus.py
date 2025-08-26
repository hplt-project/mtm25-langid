#!/usr/bin/env python3

import os
import json
from pathlib import Path
from datasets import load_dataset

def download_flores_plus():
    output_dir = Path("data/flores_plus")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading FLORES+ dataset...")
    dataset = load_dataset("openlanguagedata/flores_plus")

    for split_name in ["dev", "devtest"]:
        if split_name in dataset:
            split_data = dataset[split_name]
            print(f"Processing {split_name} split with {len(split_data)} examples...")

            processed_data = []
            for example in split_data:
                processed_data.append({
                    "text": example["text"],
                    "language": example["iso_639_3"],
                    "script": example["iso_15924"],
                    "glottocode": example["glottocode"],
                    "id": example["id"],
                    "domain": example["domain"],
                    "topic": example["topic"],
                    "url": example["url"]
                })

            output_file = output_dir / f"{split_name}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in processed_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(f"Saved {len(processed_data)} examples to {output_file}")
        else:
            print(f"Warning: {split_name} split not found in dataset")

def download_flores_plus_by_language(language_code):
    output_dir = Path("data/flores_plus")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading FLORES+ dataset for language: {language_code}")
    dataset = load_dataset("openlanguagedata/flores_plus", language_code)

    for split_name in ["dev", "devtest"]:
        if split_name in dataset:
            split_data = dataset[split_name]
            print(f"Processing {split_name} split for {language_code} with {len(split_data)} examples...")

            processed_data = []
            for example in split_data:
                processed_data.append({
                    "text": example["text"],
                    "language": example["iso_639_3"],
                    "script": example["iso_15924"],
                    "glottocode": example["glottocode"],
                    "id": example["id"],
                    "domain": example["domain"],
                    "topic": example["topic"],
                    "url": example["url"]
                })

            output_file = output_dir / f"{language_code}_{split_name}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in processed_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(f"Saved {len(processed_data)} examples to {output_file}")
        else:
            print(f"Warning: {split_name} split not found for {language_code}")

if __name__ == "__main__":
    download_flores_plus()
