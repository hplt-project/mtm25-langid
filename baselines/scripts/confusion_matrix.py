#!/usr/bin/env python3

import json
import argparse
import numpy as np


def load_confusion_matrix_from_evaluation(eval_file, languages_file=None):
    with open(eval_file, 'r') as f:
        data = json.load(f)

    confusion_dict = data['confusion_matrix']

    allowed_languages = None
    if languages_file:
        with open(languages_file, 'r') as f:
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
        .filter-control { margin: 20px 0; padding: 15px; background-color: #e9ecef; border-radius: 5px; }
        .hidden-lang { display: none; }
    </style>
</head>
<body>
    <h1>Confusion Matrices</h1>

    <ul class="nav nav-tabs" id="modelTabs" role="tablist">"""

    for i, model in enumerate(models_data.keys()):
        active = "active" if i == 0 else ""
        html += f'<li class="nav-item" role="presentation">'
        html += f'<button class="nav-link {active}" id="{model}-tab" data-bs-toggle="tab" data-bs-target="#{model}" type="button" role="tab">{model}</button>'
        html += '</li>'

    html += """    </ul>
    <div class="tab-content" id="modelTabsContent">"""

    for i, (model, (cm, languages)) in enumerate(models_data.items()):
        active = "active show" if i == 0 else ""
        html += f'<div class="tab-pane {active}" id="{model}" role="tabpanel">'

        html += f"""
        <div class="stats">
            <p><strong>Total predictions:</strong> {cm.sum()}</p>
            <p><strong>Correct predictions:</strong> {cm.trace()}</p>
            <p><strong>Accuracy:</strong> {cm.trace() / cm.sum():.4f}</p>
        </div>

        <div class="filter-control">
            <label for="errorThreshold-{model}">Hide languages with max off-diagonal errors &le; </label>
            <input type="number" id="errorThreshold-{model}" value="0" min="0" style="width: 80px;">
            <button onclick="applyFilter('{model}')" class="btn btn-sm btn-primary">Apply Filter</button>
            <button onclick="resetFilter('{model}')" class="btn btn-sm btn-secondary">Reset</button>
            <span id="hiddenCount-{model}" style="margin-left: 20px;"></span>
        </div>

        <div class="table-responsive">
            <table class="table confusion-table" id="table-{model}">
                <tr class="header-row">
                    <th></th>"""

        for lang in languages:
            html += f'<th class="lang-col" data-lang="{lang}">{lang}</th>'
        html += "<th>Total</th><th>Recall</th></tr>"

        row_totals = cm.sum(axis=1)
        col_totals = cm.sum(axis=0)

        for i, true_lang in enumerate(languages):
            row_errors = row_totals[i] - cm[i, i]
            col_errors = col_totals[i] - cm[i, i]
            max_errors = max(row_errors, col_errors)

            html += f'<tr class="lang-row" data-lang="{true_lang}" data-max-errors="{max_errors}">'
            html += f'<th class="lang-col" data-lang="{true_lang}">{true_lang}</th>'

            for j, pred_lang in enumerate(languages):
                value = cm[i, j]
                if i == j:
                    html += f'<td class="diagonal lang-col" data-lang="{pred_lang}">{value}</td>'
                elif value > 0:
                    html += f'<td class="incorrect lang-col" data-lang="{pred_lang}">{value}</td>'
                else:
                    html += f'<td class="lang-col" data-lang="{pred_lang}">{value}</td>'

            html += f'<th class="diagonal">{row_totals[i]}</th>'

            tp = cm[i, i]
            fn = row_totals[i] - tp
            recall = tp / row_totals[i] if row_totals[i] > 0 else 0
            recall_class = "bad-recall" if recall < 0.98 else "diagonal"
            html += f'<th class="{recall_class}">{recall:.3f}</th></tr>'

        html += '<tr class="totals-row"><th>Total</th>'
        for j in range(len(languages)):
            html += f'<th class="diagonal lang-col" data-lang="{languages[j]}">{col_totals[j]}</th>'
        html += f'<th class="diagonal">{cm.sum()}</th><th>-</th></tr>'

        html += '<tr class="fpr-row"><th>FPR</th>'
        for j in range(len(languages)):
            fp = col_totals[j] - cm[j, j]
            tn = cm.sum() - row_totals[j] - col_totals[j] + cm[j, j]
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fpr_class = "bad-fpr" if fpr > 0.002 else "diagonal"
            html += f'<th class="{fpr_class} lang-col" data-lang="{languages[j]}">{fpr:.3f}</th>'
        html += f'<th>-</th><th>-</th></tr>'

        html += """            </table>
        </div>
        </div>"""

    html += """    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function applyFilter(modelId) {
            const threshold = parseInt(document.getElementById('errorThreshold-' + modelId).value);
            const table = document.getElementById('table-' + modelId);
            const rows = table.querySelectorAll('.lang-row');
            const hiddenLangs = new Set();
            let totalHidden = 0;

            rows.forEach(row => {
                const maxErrors = parseInt(row.getAttribute('data-max-errors'));
                const lang = row.getAttribute('data-lang');

                if (maxErrors <= threshold) {
                    hiddenLangs.add(lang);
                    totalHidden++;
                    row.classList.add('hidden-lang');
                } else {
                    row.classList.remove('hidden-lang');
                }
            });

            table.querySelectorAll('.lang-col').forEach(cell => {
                const lang = cell.getAttribute('data-lang');
                if (hiddenLangs.has(lang)) {
                    cell.classList.add('hidden-lang');
                } else {
                    cell.classList.remove('hidden-lang');
                }
            });

            document.getElementById('hiddenCount-' + modelId).textContent =
                `Hidden: ${totalHidden} / ${rows.length} languages`;
        }

        function resetFilter(modelId) {
            document.getElementById('errorThreshold-' + modelId).value = '0';
            const table = document.getElementById('table-' + modelId);
            table.querySelectorAll('.hidden-lang').forEach(el => {
                el.classList.remove('hidden-lang');
            });
            document.getElementById('hiddenCount-' + modelId).textContent = '';
        }
    </script>
</body>
</html>"""

    with open(output_file, 'w') as f:
        f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_files", nargs="+", help="One or more evaluation JSON files")
    parser.add_argument("--languages-file")
    parser.add_argument("--output", default="confusion_matrix.html")

    args = parser.parse_args()

    models_data = {}
    for eval_file in args.evaluation_files:
        model_name = eval_file.split('/')[-1].replace('_evaluation.json', '')
        cm, languages = load_confusion_matrix_from_evaluation(eval_file, args.languages_file)
        models_data[model_name] = (cm, languages)

    create_html_confusion_matrix(models_data, args.output)
    print(f"Confusion matrices saved to {args.output}")
