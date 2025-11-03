#!/usr/bin/env python3

import argparse
import jsonlines
from collections import defaultdict


def list_disagreements(input_file, flores, models):
    disagreements_by_id = defaultdict(int)
    model1_correct_by_id = defaultdict(int)
    model2_correct_by_id = defaultdict(int)
    both_wrong_by_id = defaultdict(int)
    model1_wrong_predictions = defaultdict(lambda: defaultdict(int))
    model2_wrong_predictions = defaultdict(lambda: defaultdict(int))
    total_disagreements = 0
    total_examples = 0

    with jsonlines.open(input_file) as reader:
        for example in reader:
            total_examples += 1
            label = example["id"] if not flores else f"{example['iso_639_3']}_{example['iso_15924']}"
            predictions = example['predictions']

            if predictions[models[0]] != predictions[models[1]]:
                total_disagreements += 1
                disagreements_by_id[label] += 1

                if predictions[models[0]] == label:
                    model1_correct_by_id[label] += 1
                    model2_wrong_predictions[label][predictions[models[1]]] += 1
                elif predictions[models[1]] == label:
                    model2_correct_by_id[label] += 1
                    model1_wrong_predictions[label][predictions[models[0]]] += 1
                else:
                    both_wrong_by_id[label] += 1

    print(f"Total examples: {total_examples}")
    print(f"Total disagreements: {total_disagreements}")
    print(f"Agreement rate: {(total_examples - total_disagreements) / total_examples * 100:.2f}%")
    print(f"\nDisagreements by language ID:")
    for label, count in sorted(disagreements_by_id.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count}")

    print(f"\n{models[0]} correct (when disagreeing):")
    for label, count in sorted(model1_correct_by_id.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count}")
        for pred, pred_count in sorted(model2_wrong_predictions[label].items(), key=lambda x: x[1], reverse=True):
            print(f"    {pred}: {pred_count}")

    print(f"\n{models[1]} correct (when disagreeing):")
    for label, count in sorted(model2_correct_by_id.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count}")
        for pred, pred_count in sorted(model1_wrong_predictions[label].items(), key=lambda x: x[1], reverse=True):
            print(f"    {pred}: {pred_count}")

    print(f"\nBoth wrong (when disagreeing):")
    for label, count in sorted(both_wrong_by_id.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input merged JSONL file")
    parser.add_argument("--flores", action="store_true", help="Use iso_639_3 and iso_15924 fields instead of id")
    parser.add_argument("--models", nargs=2, metavar=("MODEL1", "MODEL2"), required=True, help="Models to compare")

    args = parser.parse_args()

    list_disagreements(args.input_file, args.flores, args.models)

