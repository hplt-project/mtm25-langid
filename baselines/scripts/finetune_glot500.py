#!/usr/bin/env python3

import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class LanguageDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label2id[label], dtype=torch.long)
        }


def load_data(data_path):
    """Load data from JSONL file"""
    print(f"Loading data from {data_path}...")

    texts = []
    labels = []

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            texts.append(data['text'])
            # Handle both 'language' and 'iso_639_3' fields
            if 'language' in data:
                # OpenLID format: "ace_Arab"
                labels.append(data['language'])
            elif 'iso_639_3' in data:
                # Flores_plus format: "ace" + "Arab" -> "ace_Arab"
                lang_code = data['iso_639_3']
                script = data.get('iso_15924', '')
                if script:
                    labels.append(f"{lang_code}_{script}")
                else:
                    labels.append(lang_code)
            else:
                raise ValueError(f"Data must contain either 'language' or 'iso_639_3' field: {data.keys()}")

    print(f"Loaded {len(texts)} examples with {len(set(labels))} unique languages")
    return texts, labels


def compute_metrics(pred):
    """Compute evaluation metrics"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Convert back to string labels
    id2label = {idx: label for label, idx in pred.label2id.items()}
    pred_labels = [id2label[pred] for pred in preds]
    true_labels = [id2label[label] for label in labels]

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro', zero_division=0
    )
    acc = accuracy_score(true_labels, pred_labels)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def finetune_glot500(train_data_path, eval_data_path, output_dir, num_epochs=3, batch_size=8):
    """Fine-tune Glot500 for language identification"""
    print("Loading Glot500 model and tokenizer...")

    # Load model and tokenizer
    model_name = "cis-lmu/glot500-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load training data
    train_texts, train_labels = load_data(train_data_path)
    eval_texts, eval_labels = load_data(eval_data_path)

    # Create datasets
    train_dataset = LanguageDataset(train_texts, train_labels, tokenizer)
    eval_dataset = LanguageDataset(eval_texts, eval_labels, tokenizer)

    # Load model for sequence classification
    num_labels = len(train_dataset.label2id)
    print(f"Number of language classes: {num_labels}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=train_dataset.id2label,
        label2id=train_dataset.label2id
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments for multi-GPU
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to=None,
        logging_steps=100,
        save_total_limit=2,
        dataloader_pin_memory=False,
        # Multi-GPU settings
        dataloader_num_workers=4,
        gradient_accumulation_steps=1,
        warmup_steps=500,
        # Distributed training
        local_rank=-1,  # Auto-detect
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Evaluating final model...")
    eval_results = trainer.evaluate()

    print("Final evaluation results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")

    # Save the model
    print(f"Saving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Save label mapping
    label_mapping = {
        'label2id': train_dataset.label2id,
        'id2label': train_dataset.id2label
    }
    with open(f"{output_dir}/label_mapping.json", 'w') as f:
        json.dump(label_mapping, f, indent=2)

    print("Fine-tuning completed!")
    return model, tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Glot500 for language identification")
    parser.add_argument("--train_data", required=True, help="Path to training data JSONL file")
    parser.add_argument("--eval_data", required=True, help="Path to evaluation data JSONL file")
    parser.add_argument("--output_dir", required=True, help="Output directory for fine-tuned model")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size per GPU")

    args = parser.parse_args()

    finetune_glot500(
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size
    )
