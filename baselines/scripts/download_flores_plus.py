#!/usr/bin/env python3

import jsonlines
from pathlib import Path
from datasets import load_dataset


def download_flores_plus(split):
    output_dir = Path("data/flores_plus")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("openlanguagedata/flores_plus", split=split)

    output_file = output_dir / f"{split}.jsonl"
    with jsonlines.open(output_file, mode='w') as writer:
        for example in dataset:
            writer.write(example)

    print(f"Saved {len(dataset)} examples to {output_file}")


if __name__ == "__main__":
    download_flores_plus("dev")
    download_flores_plus("devtest")
