#!/usr/bin/env python3

import json
import argparse
import sys
import jsonlines
import mediapipe as mp
from langcodes import Language
from eval_datasets import load_flores_data, load_udhr_data


def get_model_info(model_name):
    """Get model information for MediaPipe"""
    models = {
        "mediapipe": "mediapipe"
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")

    return models[model_name]


def convert_to_three_letter(lang_code):
    """Convert 2-letter language code to 3-letter code with script using langcodes"""
    try:
        lang = Language.get(lang_code)
        three_letter = lang.to_alpha3()

        # Get script code if available
        script = lang.script
        if script:
            return f"{three_letter}_{script}"
        else:
            return three_letter
    except:
        return lang_code


def predict_languages(dataset, model_name, split=None):
    # Get model information
    model_type = get_model_info(model_name)

    print(f"Initializing MediaPipe language detection...", file=sys.stderr)

    # Initialize MediaPipe language detection
    mp_lang = mp.solutions.language_detection
    detector = mp_lang.LanguageDetection()

    # Load data based on dataset
    if dataset == "flores":
        if split is None:
            raise ValueError("Split must be specified for FLORES+ dataset")
        data = load_flores_data(split)
    elif dataset == "udhr":
        data = load_udhr_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Available datasets: flores, udhr")

    print(f"Processing {len(data)} examples...", file=sys.stderr)
    results = []

    for i, example in enumerate(data):
        if i % 10000 == 0:
            print(f"Processed {i}/{len(data)} examples...", file=sys.stderr)

        # MediaPipe language detection
        detection = detector.detect_language(example["text"])

        if detection.language_code:
            pred_lang = convert_to_three_letter(detection.language_code)
            confidence = detection.confidence
        else:
            pred_lang = "unknown"
            confidence = 0.0

        result = example.copy()
        if "predictions" not in result:
            result["predictions"] = {}
        result["predictions"][model_name] = {
            "language": pred_lang,
            "confidence": confidence
        }
        results.append(result)

    print(f"Completed processing {len(results)} examples", file=sys.stderr)

    with jsonlines.Writer(sys.stdout) as writer:
        for result in results:
            writer.write(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MediaPipe-based language identification predictions")
    parser.add_argument("--dataset", choices=["flores", "udhr"], required=True,
                       help="Dataset to process (flores or udhr)")
    parser.add_argument("--model", choices=["mediapipe"], required=True,
                       help="Model to use (mediapipe)")
    parser.add_argument("--split", choices=["dev", "devtest"],
                       help="Data split to process (required for FLORES+ dataset)")

    args = parser.parse_args()

    # Validate split requirement for FLORES+
    if args.dataset == "flores" and args.split is None:
        parser.error("--split is required when --dataset is flores")

    predict_languages(args.dataset, args.model, args.split)
