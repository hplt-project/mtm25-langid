#!/usr/bin/env python3

import json
import argparse
import os
import numpy as np
from jinja2 import Environment, FileSystemLoader


def load_confusion_matrix_from_evaluation(eval_file, languages_file=None):
    with open(eval_file, "r") as f:
        data = json.load(f)

    confusion_dict = data["confusion_matrix"]

    allowed_languages = None
    if languages_file:
        with open(languages_file, "r") as f:
            allowed_languages = set(line.strip() for line in f if line.strip())

    if allowed_languages:
        filtered_confusion = {}
        other_row = {}

        for true_lang, predictions in confusion_dict.items():
            true_lang_mapped = true_lang if true_lang in allowed_languages else "other"

            if true_lang_mapped not in filtered_confusion:
                filtered_confusion[true_lang_mapped] = {}

            for pred_lang, count in predictions.items():
                pred_lang_mapped = pred_lang if pred_lang in allowed_languages else "other"

                if pred_lang_mapped not in filtered_confusion[true_lang_mapped]:
                    filtered_confusion[true_lang_mapped][pred_lang_mapped] = 0
                filtered_confusion[true_lang_mapped][pred_lang_mapped] += count

        confusion_dict = filtered_confusion

    languages = sorted(set(confusion_dict.keys()) |
                       set(lang for preds in confusion_dict.values() for lang in preds.keys()))

    cm = np.zeros((len(languages), len(languages)), dtype=int)
    lang_to_idx = {lang: i for i, lang in enumerate(languages)}

    for true_lang, predictions in confusion_dict.items():
        true_idx = lang_to_idx[true_lang]
        for pred_lang, count in predictions.items():
            pred_idx = lang_to_idx[pred_lang]
            cm[true_idx, pred_idx] = count

    return cm, languages


def prepare_template_data(models_data):
    template_data = {}

    for model_name, (cm, languages) in models_data.items():
        row_totals = cm.sum(axis=1)
        col_totals = cm.sum(axis=0)

        fpr_values = []
        for j in range(len(languages)):
            fp = col_totals[j] - cm[j, j]
            tn = cm.sum() - row_totals[j] - col_totals[j] + cm[j, j]
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fpr_class = "bad-fpr" if fpr > 0.002 else "diagonal"
            fpr_values.append({
                "lang": languages[j],
                "value": fpr,
                "class": fpr_class
            })

        rows = []
        for i, true_lang in enumerate(languages):
            row_errors = row_totals[i] - cm[i, i]
            col_errors = col_totals[i] - cm[i, i]
            max_errors = max(row_errors, col_errors)

            tp = cm[i, i]
            recall = tp / row_totals[i] if row_totals[i] > 0 else 0
            recall_class = "bad-recall" if recall < 0.98 else "diagonal"

            cells = []
            for j, pred_lang in enumerate(languages):
                value = cm[i, j]
                if i == j:
                    cell_class = "diagonal"
                elif value > 0:
                    cell_class = "incorrect"
                else:
                    cell_class = ""

                cells.append({
                    "lang": pred_lang,
                    "value": int(value),
                    "class": cell_class
                })

            rows.append({
                "lang": true_lang,
                "total": int(row_totals[i]),
                "recall": recall,
                "recall_class": recall_class,
                "max_errors": max_errors,
                "cells": cells
            })

        template_data[model_name] = {
            "languages": languages,
            "cm_sum": int(cm.sum()),
            "cm_trace": int(cm.trace()),
            "col_totals": [int(x) for x in col_totals],
            "fpr_values": fpr_values,
            "rows": rows
        }

    return template_data


def create_html_confusion_matrix(models_data, output_file):
    template_data = prepare_template_data(models_data)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(script_dir))
    template = env.get_template("confusion_matrix_template.html")
    html = template.render(models=template_data)
    with open(output_file, "w") as f:
        f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_files", nargs="+", help="One or more evaluation JSON files")
    parser.add_argument("--languages-file")
    parser.add_argument("--output", default="confusion_matrix.html")

    args = parser.parse_args()

    models_data = {}
    for eval_file in args.evaluation_files:
        model_name = eval_file.split("/")[-1].replace("_evaluation.json", "")
        cm, languages = load_confusion_matrix_from_evaluation(eval_file, args.languages_file)
        models_data[model_name] = (cm, languages)

    create_html_confusion_matrix(models_data, args.output)
    print(f"Confusion matrices saved to {args.output}")
