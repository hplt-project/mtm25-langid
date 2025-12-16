#!/usr/bin/env python3

import json
import sys


def load_flores_data(split):
    print(f"Loading FLORES+ {split} data...", file=sys.stderr)
    data = []

    with open(f"data/flores_plus/{split}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    print(f"Loaded {len(data)} examples", file=sys.stderr)
    return data


def load_udhr_data():
    print("Loading UDHR test data...", file=sys.stderr)
    data = []

    with open("data/udhr/test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    print(f"Loaded {len(data)} examples", file=sys.stderr)
    return data


def load_hplt_data():
    print("Loading HPLT test data...", file=sys.stderr)
    data = []

    with open("data/hplt/test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    print(f"Loaded {len(data)} examples", file=sys.stderr)
    return data
