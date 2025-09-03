#!/usr/bin/env python3

import json
import fasttext
import argparse
import sys
import jsonlines
from huggingface_hub import hf_hub_download


def load_flores_data(split):
    print(f"Loading FLORES+ {split} data...", file=sys.stderr)
    data = []

    with open(f"data/flores_plus/{split}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    print(f"Loaded {len(data)} examples", file=sys.stderr)
    return data


def load_udhr_data():
    print("Loading UDHR test data...", file=sys.stderr)
    data = []

    with open("data/udhr/test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    print(f"Loaded {len(data)} examples", file=sys.stderr)
    return data


def get_model_info(model_name):
    models = {
        "glotlid": ("cis-lmu/glotlid", "model.bin"),
        "openlid": ("laurievb/OpenLID", "model.bin"),
        "openlid-v2": ("laurievb/OpenLID-v2", "model.bin")
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")

    return models[model_name]


def predict_languages(dataset, model_name, split=None):
    repo_id, filename = get_model_info(model_name)

    print(f"Downloading {model_name} model from Hugging Face...", file=sys.stderr)
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)

    print(f"Loading model from {model_path}...", file=sys.stderr)
    model = fasttext.load_model(model_path)

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

        pred = model.predict(text_content)[0][0]
        pred_lang = pred.replace("__label__", "")

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
    parser = argparse.ArgumentParser(description="Run fastText-based language identification predictions")
    parser.add_argument("--dataset", choices=["flores", "udhr"], required=True,
                       help="Dataset to process (flores or udhr)")
    parser.add_argument("--model", choices=["glotlid", "openlid", "openlid-v2"], required=True,
                       help="Model to use (glotlid or openlid or openlid-v2)")
    parser.add_argument("--split", choices=["dev", "devtest"],
                       help="Data split to process (required for FLORES+ dataset)")

    args = parser.parse_args()

    if args.dataset == "flores" and args.split is None:
        parser.error("--split is required when --dataset is flores")

    predict_languages(args.dataset, args.model, args.split)
