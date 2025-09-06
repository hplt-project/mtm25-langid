#!/usr/bin/env python3

import json
import argparse
import sys
import jsonlines
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from eval_datasets import load_flores_data, load_udhr_data


def predict_languages(dataset, model_name, model_dir, languages_file, split=None):
    print(f"Loading {model_name} model from {model_dir}...", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    with open(languages_file, 'r') as f:
        language_labels = [line.strip() for line in f if line.strip()]
    language_labels.append("unknown")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

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

        inputs = tokenizer(
            text_content,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        if predicted_class < len(language_labels):
            pred_lang = language_labels[predicted_class]
        else:
            pred_lang = "unknown"

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
    parser = argparse.ArgumentParser(description="Run Glot500 finetuned model predictions")
    parser.add_argument("--dataset", choices=["flores", "udhr"], required=True,
                       help="Dataset to process (flores or udhr)")
    parser.add_argument("--model", required=True,
                       help="Model name for output")
    parser.add_argument("--model-dir", required=True,
                       help="Path to the finetuned model directory")
    parser.add_argument("--languages-file", required=True,
                       help="Path to the language labels file used during training")
    parser.add_argument("--split", choices=["dev", "devtest"],
                       help="Data split to process (required for FLORES+ dataset)")

    args = parser.parse_args()

    if args.dataset == "flores" and args.split is None:
        parser.error("--split is required when --dataset is flores")

    predict_languages(args.dataset, args.model, args.model_dir, args.languages_file, args.split)
