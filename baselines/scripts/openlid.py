#!/usr/bin/env python3

import json
import fasttext
import argparse
import sys
import jsonlines
from huggingface_hub import hf_hub_download


def load_data(split):
    print(f"Loading {split} data...", file=sys.stderr)
    data = []

    with open(f"data/flores_plus/{split}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    print(f"Loaded {len(data)} examples", file=sys.stderr)
    return data


def predict_languages(split):
    print("Downloading OpenLID model from Hugging Face...", file=sys.stderr)
    model_path = hf_hub_download(repo_id="laurievb/OpenLID", filename="model.bin")

    print(f"Loading model from {model_path}...", file=sys.stderr)
    model = fasttext.load_model(model_path)

    data = load_data(split)

    print(f"Processing {len(data)} examples...", file=sys.stderr)
    results = []

    for i, example in enumerate(data):
        if i % 10000 == 0:
            print(f"Processed {i}/{len(data)} examples...", file=sys.stderr)

        pred = model.predict(example["text"])[0][0]
        pred_lang = pred.replace("__label__", "")

        result = example.copy()
        if "predictions" not in result:
            result["predictions"] = {}
        result["predictions"]["openlid"] = pred_lang
        results.append(result)

    print(f"Completed processing {len(results)} examples", file=sys.stderr)

    with jsonlines.Writer(sys.stdout) as writer:
        for result in results:
            writer.write(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenLID predictions on flores-plus dataset")
    parser.add_argument("split", choices=["dev", "devtest"], help="Data split to process")

    args = parser.parse_args()
    predict_languages(args.split)
