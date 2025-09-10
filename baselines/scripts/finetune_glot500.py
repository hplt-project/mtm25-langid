#!/usr/bin/env python3

import os
import torch
import random
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import flash_attn
from gradual_unfreezing import GradualUnfreezingCallback


def load_language_list(languages_file_path):
    with open(languages_file_path, 'r') as f:
        language_labels = [line.strip() for line in f if line.strip()]
    return language_labels


def preprocess_flores(examples, tokenizer, label2id):
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding=False,
        return_tensors=None
    )
    language_codes = [f"{iso_639_3}_{iso_15924}" for iso_639_3, iso_15924 in zip(examples['iso_639_3'], examples['iso_15924'])]

    labels = []
    for language_code in language_codes:
        if language_code in label2id:
            labels.append(label2id[language_code])
        else:
            labels.append(label2id["unknown"])

    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': labels
    }


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)

    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


def create_compute_metrics_with_tasting(eval_dataset, tokenizer, language_labels,num_samples=5, seed=42):
    random.seed(seed)
    total_samples = len(eval_dataset)
    selected_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    selected_indices.sort()

    def compute_metrics_with_tasting(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        print("\n" + "="*50)
        print("Tasting evaluation dataset with current model predictions:")
        print("="*50)
        taste_dataset(eval_dataset, tokenizer, language_labels, preds, selected_indices)
        print("="*50 + "\n")

        return compute_metrics(pred)

    return compute_metrics_with_tasting


def taste_dataset(dataset, tokenizer, language_labels, predictions=None, sample_indices=None):
    if sample_indices is None:
        sample_indices = list(range(min(5, len(dataset))))

    for idx, i in enumerate(sample_indices):
        print(f"Sample {idx+1} (index {i}):")

        tokens = tokenizer.convert_ids_to_tokens(dataset[i]['input_ids'])
        text = tokenizer.decode(dataset[i]['input_ids'], skip_special_tokens=True)

        print(f"  text: {text}")
        print(f"  tokens: {tokens[:40]}...")

        if dataset[i]['labels'] < len(language_labels):
            print(f"  label: {language_labels[dataset[i]['labels']]}")
        else:
            print(f"  label: unknown")

        if predictions is not None:
            if predictions[i] < len(language_labels):
                print(f"  prediction: {language_labels[predictions[i]]}")
            else:
                print(f"  prediction: unknown")

        print()


def finetune_glot500(train_path, eval_data_path, output_dir, languages_file_path, num_epochs=3, batch_size=32, max_samples=None, gradual_unfreezing=False):
    tokenizer = AutoTokenizer.from_pretrained("cis-lmu/glot500-base")

    language_labels = load_language_list(languages_file_path)

    print(f"Loading preprocessed training data from {train_path}...")
    train_dataset = Dataset.load_from_disk(train_path)
    if max_samples is not None and max_samples < len(train_dataset):
        train_dataset = train_dataset.select(range(max_samples))
        print(f"Using {max_samples} samples from preprocessed training data")
    print("Training dataset:")
    taste_dataset(train_dataset, tokenizer, language_labels)

    label2id = {label: idx for idx, label in enumerate(language_labels)}
    label2id["unknown"] = len(label2id)

    eval_dataset = Dataset.from_json(eval_data_path)
    eval_dataset = eval_dataset.map(
        lambda examples: preprocess_flores(examples, tokenizer, label2id),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    print("Evaluation dataset:")
    taste_dataset(eval_dataset, tokenizer, language_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        "cis-lmu/glot500-base",
        num_labels=len(label2id),
        device_map="auto"
    )

    callbacks = []
    if gradual_unfreezing:
        unfreezing_schedule = {  # this is number of examples before dividing into batches
            0: 0,
            10_000_000: 2,
            20_000_000: 4,
            30_000_000: 8,
            40_000_000: 12, # glot500 is based on XLM-R base, which has 12 layers
        }

        print("Gradual unfreezing enabled")

        callbacks.append(
            GradualUnfreezingCallback(
            model=model,
            unfreezing_schedule=unfreezing_schedule,
        ))

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",
        eval_steps=2000,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=2e-5,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=create_compute_metrics_with_tasting(eval_dataset, tokenizer, language_labels, num_samples=5),
        callbacks=callbacks
    )

    trainer.train()

    final_results = trainer.evaluate()
    print(f"Final evaluation results: {final_results}")

    trainer.save_model()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Glot500 for language identification")
    parser.add_argument("--train-path", required=True, help="Path to preprocessed training data directory")
    parser.add_argument("--eval-data", required=True, help="Path to evaluation data JSONL file")
    parser.add_argument("--output-dir", required=True, help="Output directory for fine-tuned model")
    parser.add_argument("--languages-file", required=True, help="Path to languages.txt file")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size per GPU")
    parser.add_argument("--gradual-unfreezing", action="store_true", help="Enable gradual unfreezing of transformer layers")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples from training data (default: use all)")

    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    finetune_glot500(train_path=args.train_path, eval_data_path=args.eval_data, output_dir=args.output_dir,
                     languages_file_path=args.languages_file, num_epochs=args.num_epochs, batch_size=args.batch_size,
                     max_samples=args.max_samples, gradual_unfreezing=args.gradual_unfreezing)
