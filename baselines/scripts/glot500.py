#!/usr/bin/env python3

import json
import argparse
import sys
import jsonlines
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
import torch


def load_data(split):
    print(f"Loading {split} data...", file=sys.stderr)
    data = []

    with open(f"data/flores_plus/{split}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    print(f"Loaded {len(data)} examples", file=sys.stderr)
    return data


def predict_languages(split, model_path=None):
    if model_path:
        print(f"Loading fine-tuned Glot500 model from {model_path}...", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Load label mapping
        with open(f"{model_path}/label_mapping.json", 'r') as f:
            label_mapping = json.load(f)
        id2label = label_mapping['id2label']

    else:
        print("Loading base Glot500 model from Hugging Face...", file=sys.stderr)
        # Load tokenizer and model
        model_name = "cis-lmu/glot500-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        id2label = None

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    data = load_data(split)

    print(f"Processing {len(data)} examples...", file=sys.stderr)
    results = []

    for i, example in enumerate(data):
        if i % 10000 == 0:
            print(f"Processed {i}/{len(data)} examples...", file=sys.stderr)

        text = example["text"]

        # Tokenize input
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        # Convert prediction to label
        if id2label:
            pred_lang = id2label[str(predicted_class)]
        else:
            # Fallback for base model
            pred_lang = f"lang_{predicted_class}"

        result = example.copy()
        if "predictions" not in result:
            result["predictions"] = {}
        result["predictions"]["glot500"] = pred_lang
        results.append(result)

    print(f"Completed processing {len(results)} examples", file=sys.stderr)

    with jsonlines.Writer(sys.stdout) as writer:
        for result in results:
            writer.write(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Glot500 predictions on flores-plus dataset")
    parser.add_argument("split", choices=["dev", "devtest"], help="Data split to process")
    parser.add_argument("--model_path", help="Path to fine-tuned model directory (optional)")

    args = parser.parse_args()
    predict_languages(args.split, args.model_path)
