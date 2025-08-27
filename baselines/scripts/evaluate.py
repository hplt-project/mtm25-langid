#!/usr/bin/env python3

import json
import argparse
import sys
from collections import defaultdict


def calculate_metrics(y_true, y_pred, unique_labels):
    f1_scores = {}
    per_language_metrics = {}

    for label in unique_labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t != label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        f1_scores[label] = f1
        per_language_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }

    macro_f1 = sum(f1_scores.values()) / len(f1_scores) if f1_scores else 0

    return per_language_metrics, macro_f1


def evaluate_predictions(input_file, model):
    data = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    print(f"Loaded {len(data)} examples", file=sys.stderr)

    correct = 0
    total = len(data)
    y_true = []
    y_pred = []
    lang_counts = defaultdict(int)

    for example in data:
        true_lang = f"{example['language']}_{example['script']}"
        pred_lang = example['predictions'][model]

        if pred_lang == true_lang:
            correct += 1

        y_true.append(true_lang)
        y_pred.append(pred_lang)
        lang_counts[true_lang] += 1

    accuracy = correct / total

    #unique_labels = sorted(set(y_true + y_pred))  # TODO is this correct?
    unique_labels = sorted(set(y_true))
    per_lang_metrics, macro_f1 = calculate_metrics(y_true, y_pred, unique_labels)

    for lang in per_lang_metrics:
        per_lang_metrics[lang]["count"] = lang_counts[lang]

    results = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "total_examples": total,
        "per_language_metrics": per_lang_metrics
    }

    json.dump(results, sys.stdout, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate language identification predictions")
    parser.add_argument("input_file", help="Input JSONL file with predictions")
    parser.add_argument("--model", default="glotlid", help="Model name to evaluate (default: glotlid)")

    args = parser.parse_args()
    evaluate_predictions(args.input_file, args.model)
