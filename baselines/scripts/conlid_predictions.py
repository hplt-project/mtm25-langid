#!/usr/bin/env python3

import json
import argparse
import sys
import jsonlines
import os
from eval_datasets import load_flores_data, load_udhr_data

from model import ConLID


def predict_languages(dataset, model_name, model_dir, split=None):

    print(f"Loading {model_name} model from {model_dir}...", file=sys.stderr)
    model = ConLID.from_pretrained(model_dir)

    if dataset == "flores":
        if split is None:
            raise ValueError("Split must be specified for FLORES+ dataset")
        data = load_flores_data(split)
    elif dataset == "udhr":
        data = load_udhr_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Available datasets: flores, udhr")

    print(f"Processing {len(data)} examples...", file=sys.stderr)
    results = []

    for i, example in enumerate(data):
        if i % 10000 == 0:
            print(f"Processed {i}/{len(data)} examples...", file=sys.stderr)

        if dataset == "flores":
            text_content = example["text"]
        elif dataset == "udhr":
            text_content = example["sentence"]

        # Get top prediction
        predictions, probabilities = model.predict(text_content, k=1)
        pred_lang = predictions[0]

        result = example.copy()
        if "predictions" not in result:
            result["predictions"] = {}
        result["predictions"][model_name] = pred_lang
        results.append(result)

    print(f"Completed processing {len(results)} examples", file=sys.stderr)

    with jsonlines.Writer(sys.stdout) as writer:
        for result in results:
            writer.write(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ConLID-based language identification predictions")
    parser.add_argument("--dataset", choices=["flores", "udhr"], required=True,
                       help="Dataset to process (flores or udhr)")
    parser.add_argument("--model", choices=["conlid"], required=True,
                       help="Model to use (conlid)")
    parser.add_argument("--model-dir", required=True,
                       help="Path to the model directory")
    parser.add_argument("--split", choices=["dev", "devtest"],
                       help="Data split to process (required for FLORES+ dataset)")

    args = parser.parse_args()

    if args.dataset == "flores" and args.split is None:
        parser.error("--split is required when --dataset is flores")

    predict_languages(args.dataset, args.model, args.model_dir, args.split)
