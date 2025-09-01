#!/usr/bin/env python3

import jsonlines
from pathlib import Path
from datasets import load_dataset


def download_openlid():
    output_dir = Path("data/OpenLID-v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("laurievb/OpenLID-v2", split="train")

    output_file = output_dir / "train.jsonl"
    with jsonlines.open(output_file, mode='w') as writer:
        for example in dataset:
            writer.write(example)

    print(f"Saved {len(dataset)} examples to {output_file}")


if __name__ == "__main__":
    download_openlid()
