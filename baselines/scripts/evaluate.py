#!/usr/bin/env python3

import json
import argparse
import sys
import jsonlines
from collections import defaultdict, Counter
from dataclasses import dataclass


def extract_language(example, dataset_type):
    if dataset_type == "flores":
        return f"{example['iso_639_3']}_{example['iso_15924']}"
    elif dataset_type == "udhr":
        return example['id']
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def load_languages_file(languages_file):
    with open(languages_file, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def load_data(input_file, model, dataset_type):
    y_true = []
    y_pred = []

    with jsonlines.open(input_file) as reader:
        for example in reader:
            true_lang = extract_language(example, dataset_type)
            pred_lang = example['predictions'][model]

            y_true.append(true_lang)
            y_pred.append(pred_lang)

    return y_true, y_pred


def confusion_matrix(golds, preds):
    cm = defaultdict(Counter)
    for gold, pred in zip(golds, preds):
        cm[gold][pred] += 1
    return cm


def calculate_metrics(confusion_matrix, allowed_languages):
    per_langauge = {}

    valid_decisions = 0  # all decisions done on supported target languages
    for gold in confusion_matrix:
        if gold not in allowed_languages:
            continue

        for pred in confusion_matrix[gold]:
            valid_decisions += confusion_matrix[gold][pred]

    for gold in confusion_matrix:
        if gold not in allowed_languages:
            continue

        tp = confusion_matrix[gold][gold]
        fn = sum(confusion_matrix[gold][lang] for lang in confusion_matrix[gold] if lang != gold)  # punish all false negatives when gold is supported
        fp = sum(confusion_matrix[lang][gold] for lang in confusion_matrix if lang != gold and lang in allowed_languages)  # do not punish false positives when the target language is not supported

        real_negatives = valid_decisions - tp - fn
        fpr = fp / real_negatives if real_negatives > 0 else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_langauge[gold] = {"f1": f1, "tp": tp, "fn": fn, "fp": fp, "fpr": fpr, "precision": precision, "recall": recall}

    return per_langauge


def evaluate_predictions(input_file, model, dataset_type, languages_file=None):
    allowed_languages = load_languages_file(languages_file) if languages_file else None
    golds, preds = load_data(input_file, model, dataset_type)

    cm = confusion_matrix(golds, preds)
    per_lang_metrics = calculate_metrics(cm, allowed_languages)

    macro_f1 = sum(per_lang_metrics[lang]["f1"] for lang in per_lang_metrics) / len(per_lang_metrics)
    macro_fpr = sum(per_lang_metrics[lang]["fpr"] for lang in per_lang_metrics) / len(per_lang_metrics)
    macro_precision = sum(per_lang_metrics[lang]["precision"] for lang in per_lang_metrics) / len(per_lang_metrics)
    macro_recall = sum(per_lang_metrics[lang]["recall"] for lang in per_lang_metrics) / len(per_lang_metrics)
    num_gold_examples = sum(per_lang_metrics[lang]["tp"] + per_lang_metrics[lang]["fn"] for lang in per_lang_metrics)

    results = {
        "macro_averages": {
            "f1": macro_f1,
            "fpr": macro_fpr,
            "precision": macro_precision,
            "recall": macro_recall,
            "num_examples": num_gold_examples
        },
        "per_language": per_lang_metrics,
        "confusion_matrix": cm
    }

    json.dump(results, sys.stdout, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate language identification predictions")
    parser.add_argument("input_file", help="Input JSONL file with predictions")
    parser.add_argument("--model", required=True, help="Model name to evaluate")
    parser.add_argument("--languages-file", required=True, help="File containing list of languages to restrict evaluation to (one per line)")
    parser.add_argument("--dataset", choices=["flores", "udhr"], required=True, help="Dataset type (flores or udhr)")


    args = parser.parse_args()
    evaluate_predictions(args.input_file, args.model, args.dataset, args.languages_file)
