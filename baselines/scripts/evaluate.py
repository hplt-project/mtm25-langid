#!/usr/bin/env python3

import json
import argparse
import sys
import jsonlines
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class AggregatedMetrics:
    languages: list
    accuracy: float
    macro_f1: float
    macro_fpr: float
    num_examples: int


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
    lang_counts = defaultdict(int)

    with jsonlines.open(input_file) as reader:
        for example in reader:
            true_lang = extract_language(example, dataset_type)
            pred_lang = example['predictions'][model]

            y_true.append(true_lang)
            y_pred.append(pred_lang)
            lang_counts[true_lang] += 1

    return y_true, y_pred, lang_counts


def gather_confusion_data(y_true, y_pred, unique_labels):
    confusion_data = {lang: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for lang in unique_labels}
    for t, p in zip(y_true, y_pred):
        for lang in unique_labels:
            if t == lang and p == lang:
                confusion_data[lang]["tp"] += 1
            elif t != lang and p == lang:
                confusion_data[lang]["fp"] += 1
            elif t == lang and p != lang:
                confusion_data[lang]["fn"] += 1
            else:
                confusion_data[lang]["tn"] += 1
    return confusion_data


def calculate_metrics(confusion_data):
    per_language_metrics = {}

    for lang in confusion_data:
        d = confusion_data[lang]

        precision = d["tp"] / (d["tp"] + d["fp"]) if (d["tp"] + d["fp"]) > 0 else 0
        recall = d["tp"] / (d["tp"] + d["fn"]) if (d["tp"] + d["fn"]) > 0 else 0
        fpr = d["fp"] / (d["fp"] + d["tn"]) if (d["fp"] + d["tn"]) > 0 else 0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_language_metrics[lang] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            **d
        }

    return per_language_metrics


def aggregate_metrics(per_lang_metrics, allowed_languages=None):
    if allowed_languages is None:
        languages = sorted(per_lang_metrics.keys())
        relevant_metrics = per_lang_metrics
    else:
        languages = sorted(allowed_languages)
        relevant_metrics = {lang: per_lang_metrics[lang] for lang in allowed_languages if lang in per_lang_metrics}

    if not relevant_metrics:
        return AggregatedMetrics(languages=languages, accuracy=0.0, macro_f1=0.0, macro_fpr=0.0, num_examples=0)

    total_correct = sum(metrics["tp"] for metrics in relevant_metrics.values())
    total_evaluated = sum(metrics["count"] for metrics in relevant_metrics.values())

    accuracy = total_correct / total_evaluated if total_evaluated > 0 else 0
    macro_f1 = sum(metrics["f1"] for metrics in relevant_metrics.values()) / len(relevant_metrics) if relevant_metrics else 0
    macro_fpr = sum(metrics["fpr"] for metrics in relevant_metrics.values()) / len(relevant_metrics) if relevant_metrics else 0

    return AggregatedMetrics(
        languages=languages,
        accuracy=accuracy,
        macro_f1=macro_f1,
        macro_fpr=macro_fpr,
        num_examples=total_evaluated
    )


def evaluate_predictions(input_file, model, dataset_type, languages_file=None):
    allowed_languages = load_languages_file(languages_file) if languages_file else None
    y_true, y_pred, lang_counts = load_data(input_file, model, dataset_type)

    unique_labels = sorted(set(y_true))
    confusion_data = gather_confusion_data(y_true, y_pred, unique_labels)
    per_lang_metrics = calculate_metrics(confusion_data)

    for lang in per_lang_metrics:
        per_lang_metrics[lang]["count"] = lang_counts[lang]

    aggregated = []
    if allowed_languages is not None:
        metrics = aggregate_metrics(per_lang_metrics, allowed_languages)
        aggregated.append({
            "languages": metrics.languages,
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "macro_fpr": metrics.macro_fpr,
            "num_examples": metrics.num_examples
        })

    all_metrics = aggregate_metrics(per_lang_metrics)
    all_metrics_dict = {
        "accuracy": all_metrics.accuracy,
        "macro_f1": all_metrics.macro_f1,
        "macro_fpr": all_metrics.macro_fpr,
        "num_examples": all_metrics.num_examples
    }

    results = {
        "all": all_metrics_dict,
        "aggregated": aggregated,
        "per_language": per_lang_metrics
    }

    json.dump(results, sys.stdout, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate language identification predictions")
    parser.add_argument("input_file", help="Input JSONL file with predictions")
    parser.add_argument("--model", choices=["glotlid", "openlid", "glot500"], default="glotlid", help="Model name to evaluate (default: glotlid)")
    parser.add_argument("--dataset", choices=["flores", "udhr"], required=True, help="Dataset type (flores or udhr)")
    parser.add_argument("--languages-file", help="Optional file containing list of languages to restrict evaluation to (one per line)")

    args = parser.parse_args()
    evaluate_predictions(args.input_file, args.model, args.dataset, args.languages_file)
