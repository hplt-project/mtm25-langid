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


def evaluate_model(split):
    print("Downloading GlotLID model from Hugging Face...", file=sys.stderr)
    model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")

    print(f"Loading model from {model_path}...", file=sys.stderr)
    model = fasttext.load_model(model_path)

    data = load_data(split)

    print(f"Evaluating {len(data)} examples...", file=sys.stderr)
    correct = 0
    total = len(data)
    results = []

    for i, example in enumerate(data):
        if i % 1000 == 0:
            print(f"Processed {i}/{total} examples...", file=sys.stderr)

        pred = model.predict(example["text"])[0][0]
        pred_lang = pred.replace("__label__", "")
        true_lang = example["language"]

        if pred_lang == true_lang:
            correct += 1

        results.append({
            "id": example["id"],
            "pred_glotlid": pred_lang
        })

    accuracy = correct / total
    print(f"Accuracy: {correct}/{total} = {accuracy:.3f} ({accuracy*100:.1f}%)", file=sys.stderr)

    with jsonlines.Writer(sys.stdout) as writer:
        for result in results:
            writer.write(result)

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GlotLID on flores-plus dataset")
    parser.add_argument("split", choices=["dev", "devtest"], help="Data split to evaluate")

    args = parser.parse_args()

    evaluate_model(args.split)
