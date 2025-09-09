#!/usr/bin/env python3

import argparse
from datasets import Dataset
from transformers import AutoTokenizer


def load_language_list(languages_file_path):
    with open(languages_file_path, 'r') as f:
        language_labels = [line.strip() for line in f if line.strip()]
    return language_labels


def preprocess_openlid(examples, tokenizer, language_labels):
    label2id = {label: idx for idx, label in enumerate(language_labels)}
    label2id["unknown"] = len(label2id)

    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding=False,
        return_tensors=None
    )

    labels = []
    for language in examples['language']:
        if language in label2id:
            labels.append(label2id[language])
        else:
            labels.append(label2id["unknown"])

    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': labels
    }


def preprocess_openlid_data(input_file, output_dir, languages_file, batch_size=10000, max_samples=None):
    tokenizer = AutoTokenizer.from_pretrained("cis-lmu/glot500-base")
    language_labels = load_language_list(languages_file)
    
    print("Loading dataset...")
    dataset = Dataset.from_json(input_file)
    
    print("Filtering...")
    def is_valid_row(example, idx):
        if example['text'] is None:
            print(f"Filtering out line {idx}: text is None")
            return False
        if example['language'] is None:
            print(f"Filtering out line {idx}: language is None")
            return False
        return True
    
    dataset = dataset.filter(is_valid_row, with_indices=True)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Using {max_samples} samples from training data")
    
    print("Shuffling...")
    dataset = dataset.shuffle()
    
    print("Tokenizing...")
    dataset = dataset.map(
        lambda examples: preprocess_openlid(examples, tokenizer, language_labels),
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names
    )
    
    print(f"Saving to {output_dir}...")
    dataset.save_to_disk(output_dir)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess OpenLID data")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output arrow directory")
    parser.add_argument("--languages-file", required=True, help="Languages file")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for processing")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    
    args = parser.parse_args()
    
    preprocess_openlid_data(args.input, args.output, args.languages_file, args.batch_size, args.max_samples)
