#!/usr/bin/env python3

import argparse
import jsonlines
from collections import defaultdict
from eval_datasets import load_flores_data, load_udhr_data


def main(dataset_type, split, predictions_list, output_file):

    if dataset_type == "flores":
        dataset = load_flores_data(split)
    elif dataset_type == "udhr":
        dataset = load_udhr_data()

    predictions = defaultdict(list)

    for model_name, pred_file in predictions_list:
        with open(pred_file, 'r', encoding='utf-8') as f:
            for line in f:
                predictions[model_name].append(line.strip())

        if len(predictions[model_name]) != len(dataset):
            raise ValueError(f"Prediction file {pred_file} has {len(predictions[model_name])} lines but dataset has {len(dataset)} examples")

    with jsonlines.open(output_file, 'w') as writer:
        for i, example in enumerate(dataset):
            output_example = example.copy()
            output_example['predictions'] = {}

            for model_name in predictions:
                output_example['predictions'][model_name] = predictions[model_name][i]

            writer.write(output_example)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["flores", "udhr"], required=True)
    parser.add_argument("--split", choices=["dev", "devtest"])
    parser.add_argument("--prediction", nargs=2, metavar=("MODEL", "FILE"), dest="predictions", action="append", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.dataset == "flores" and not args.split:
        parser.error("--split is required for flores dataset")

    main(args.dataset, args.split, args.predictions, args.output)
