#!/usr/bin/env python3

import json
import argparse
import jsonlines
import numpy as np
from sklearn.metrics import confusion_matrix


def extract_language(example):
    if 'iso_639_3' in example and 'iso_15924' in example:
        return f"{example['iso_639_3']}_{example['iso_15924']}"
    elif 'id' in example:
        return example['id']
    else:
        raise ValueError("Cannot determine language from example")


def load_predictions_data(jsonl_file, languages_file=None):
    models_data = {}
    allowed_languages = None

    if languages_file:
        with open(languages_file, 'r') as f:
            allowed_languages = set(line.strip() for line in f if line.strip())

    with jsonlines.open(jsonl_file) as reader:
        for example in reader:
            true_lang = extract_language(example)

            if allowed_languages is None:
                true_lang_mapped = true_lang
            else:
                true_lang_mapped = true_lang if true_lang in allowed_languages else "other"

            for model, pred_lang in example['predictions'].items():
                if model not in models_data:
                    models_data[model] = {'y_true': [], 'y_pred': []}

                if allowed_languages is None:
                    pred_lang_mapped = pred_lang
                else:
                    pred_lang_mapped = pred_lang if pred_lang in allowed_languages else "other"

                models_data[model]['y_true'].append(true_lang_mapped)
                models_data[model]['y_pred'].append(pred_lang_mapped)

    return models_data


def create_html_confusion_matrix(models_data, output_file):
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Confusion Matrices</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .confusion-table { font-size: 10px; }
        .confusion-table th, .confusion-table td {
            border: 1px solid #ddd !important;
            padding: 2.4px !important;
            text-align: center;
            font-size: 9px;
            white-space: normal !important;
            word-wrap: break-word !important;
        }
        .confusion-table th { background-color: #f2f2f2 !important; font-weight: bold; }
        .confusion-table td.diagonal { background-color: #d4edda !important; }
        .confusion-table th.diagonal { background-color: #d4edda !important; }
        .confusion-table td.incorrect { background-color: #f8d7da !important; }
        .confusion-table th.bad-recall { background-color: #f8d7da !important; }
        .confusion-table th.bad-fpr { background-color: #f8d7da !important; }
        .stats { margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px; font-size: 14px; }
        .nav-tabs .nav-link { font-size: 14px; }
        .tab-pane { transition: none !important; }
        .tab-pane.fade { transition: none !important; }
    </style>
</head>
<body>
    <h1>Confusion Matrices</h1>

    <ul class="nav nav-tabs" id="modelTabs" role="tablist">"""

    # Create tab headers
    for i, model in enumerate(models_data.keys()):
        active = "active" if i == 0 else ""
        html += f'<li class="nav-item" role="presentation">'
        html += f'<button class="nav-link {active}" id="{model}-tab" data-bs-toggle="tab" data-bs-target="#{model}" type="button" role="tab">{model}</button>'
        html += '</li>'

    html += """    </ul>
    <div class="tab-content" id="modelTabsContent">"""

    # Create tab content for each model
    for i, (model, data) in enumerate(models_data.items()):
        y_true, y_pred = data['y_true'], data['y_pred']
        languages = sorted(set(y_true + y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=languages)

        active = "active show" if i == 0 else ""
        html += f'<div class="tab-pane {active}" id="{model}" role="tabpanel">'

        html += f"""
        <div class="stats">
            <p><strong>Total predictions:</strong> {cm.sum()}</p>
            <p><strong>Correct predictions:</strong> {cm.trace()}</p>
            <p><strong>Accuracy:</strong> {cm.trace() / cm.sum():.4f}</p>
        </div>

        <div class="table-responsive">
            <table class="table confusion-table">
                <tr>
                    <th></th>"""

        # Header row
        for lang in languages:
            html += f"<th>{lang}</th>"
        html += "<th>Total</th><th>Recall</th></tr>"

        # Data rows
        row_totals = cm.sum(axis=1)
        col_totals = cm.sum(axis=0)

        for i, true_lang in enumerate(languages):
            html += f"<tr><th>{true_lang}</th>"

            for j, pred_lang in enumerate(languages):
                value = cm[i, j]
                if i == j:
                    html += f'<td class="diagonal">{value}</td>'
                elif value > 0:
                    html += f'<td class="incorrect">{value}</td>'
                else:
                    html += f'<td>{value}</td>'

            # Row total
            html += f'<th class="diagonal">{row_totals[i]}</th>'

            # Recall (TP / (TP + FN))
            tp = cm[i, i]
            fn = row_totals[i] - tp
            recall = tp / row_totals[i] if row_totals[i] > 0 else 0
            recall_class = "bad-recall" if recall < 0.98 else "diagonal"
            html += f'<th class="{recall_class}">{recall:.3f}</th></tr>'

        # Column totals row
        html += '<tr><th>Total</th>'
        for j in range(len(languages)):
            html += f'<th class="diagonal">{col_totals[j]}</th>'
        html += f'<th class="diagonal">{cm.sum()}</th><th>-</th></tr>'

        # FPR row
        html += '<tr><th>FPR</th>'
        for j in range(len(languages)):
            # FPR (FP / (FP + TN))
            fp = col_totals[j] - cm[j, j]
            tn = cm.sum() - row_totals[j] - col_totals[j] + cm[j, j]
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fpr_class = "bad-fpr" if fpr > 0.002 else "diagonal"
            html += f'<th class="{fpr_class}">{fpr:.3f}</th>'
        html += f'<th>-</th><th>-</th></tr>'

        html += """            </table>
        </div>
        </div>"""

    html += """    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""

    with open(output_file, 'w') as f:
        f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_file")
    parser.add_argument("--languages-file")
    parser.add_argument("--output", default="confusion_matrix.html")

    args = parser.parse_args()

    models_data = load_predictions_data(args.predictions_file, args.languages_file)

    create_html_confusion_matrix(models_data, args.output)
    print(f"Confusion matrices saved to {args.output}")
