#!/usr/bin/env python3

import argparse
import jsonlines
import sys


def get_content_hash(example):
    items = {k: v for k, v in example.items() if k != 'predictions'}
    return str(hash(str(sorted(items.items()))))


def merge_predictions_files(input_files, output_file):
    if output_file:
        writer = jsonlines.open(output_file, 'w')
    else:
        writer = jsonlines.Writer(sys.stdout)
    readers = [jsonlines.open(f) for f in input_files]

    line_num = 0
    for examples in zip(*readers):
        reference_key = get_content_hash(examples[0])

        merged_example = examples[0].copy()
        merged_example['predictions'] = {}

        for i, example in enumerate(examples):
            example_key = get_content_hash(example)

            if example_key != reference_key:
                print(f"ERROR: Content mismatch at line {line_num}")
                print(f"File: {input_files[i]}")
                exit(1)

            for model, prediction in example['predictions'].items():
                merged_example['predictions'][model] = prediction

        writer.write(merged_example)
        line_num += 1

    if output_file:
        writer.close()
    for reader in readers:
        reader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", nargs="+", help="Input prediction JSONL files to merge")
    parser.add_argument("--output", help="Output merged file (default: stdout)")

    args = parser.parse_args()

    merge_predictions_files(args.input_files, args.output)
