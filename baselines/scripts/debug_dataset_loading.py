#!/usr/bin/env python3

import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import flash_attn

# Global variables for line tracking
current_batch_start = 0
batch_size = 1000  # Default batch size for datasets


def load_language_list(languages_file_path):
    with open(languages_file_path, 'r') as f:
        language_labels = [line.strip() for line in f if line.strip()]
    return language_labels


def preprocess_openlid(examples, tokenizer, language_labels):
    """Preprocess OpenLID dataset examples."""
    global current_batch_start, batch_size

    # Create label mapping
    label2id = {label: idx for idx, label in enumerate(language_labels)}
    label2id["unknown"] = len(label2id)

    # Calculate line numbers for this batch
    batch_size = len(examples['text'])
    line_start = current_batch_start
    line_end = current_batch_start + batch_size - 1
    current_batch_start += batch_size

    # Tokenize texts
    try:
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding=False,  # We'll use DataCollatorWithPadding
            return_tensors=None  # Return lists, not tensors
        )
    except Exception as e:
        print(f"Tokenization failed at lines {line_start}-{line_end}: {str(e)}")
        print(f"Text samples that failed:")
        for i, text in enumerate(examples['text']):
            line_num = line_start + i
            print(f"  Line {line_num}: {repr(text)}")
        raise

    # Map labels
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


def debug_dataset_loading(train_data_path, languages_file_path):
    tokenizer = AutoTokenizer.from_pretrained("cis-lmu/glot500-base")
    language_labels = load_language_list(languages_file_path)

    print(f"Loading dataset from {train_data_path}")
    train_dataset = Dataset.from_json(train_data_path)

    print(f"Dataset loaded with {len(train_dataset)} samples")
    print(f"Dataset columns: {train_dataset.column_names}")

    # Filter out rows with null text or language
    print("Filtering out rows with null text or language...")
    def is_valid_row(example, idx):
        if example['text'] is None:
            print(f"Filtering out line {idx}: text is None")
            return False
        if example['language'] is None:
            print(f"Filtering out line {idx}: language is None")
            return False
        return True

    original_size = len(train_dataset)
    train_dataset = train_dataset.filter(is_valid_row, with_indices=True)
    filtered_size = len(train_dataset)
    print(f"Filtered out {original_size - filtered_size} invalid rows. Remaining: {filtered_size}")

    print("Starting preprocessing...")
    train_dataset = train_dataset.map(
        lambda examples: preprocess_openlid(examples, tokenizer, language_labels),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug dataset loading")
    parser.add_argument("--train-data", required=True, help="Path to training data JSONL file")
    parser.add_argument("--languages-file", required=True, help="Path to languages.txt file")

    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    debug_dataset_loading(train_data_path=args.train_data, languages_file_path=args.languages_file)
