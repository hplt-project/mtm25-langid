#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import flash_attn


def load_language_list(languages_file_path):
    with open(languages_file_path, 'r') as f:
        language_labels = [line.strip() for line in f if line.strip()]
    return language_labels


class OpenLIDDataset(TorchDataset):
    def __init__(self, data_path, tokenizer, language_labels):
        self.tokenizer = tokenizer
        self.language_labels = language_labels
        self.label2id = {label: idx for idx, label in enumerate(language_labels)}
        self.label2id["unknown"] = len(self.label2id)

        self.hf_dataset = Dataset.from_json(data_path)

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        encodings = self.tokenizer(
            item['text'],
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

        language = item.get('language')
        if language not in self.label2id:
            label = self.label2id["unknown"]
        else:
            label = self.label2id[language]

        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class FloresDataset(TorchDataset):
    def __init__(self, data_path, tokenizer, label2id):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.hf_dataset = Dataset.from_json(data_path)

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        encodings = self.tokenizer(
            item['text'],
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

        language = f"{item['iso_639_3']}_{item['iso_15924']}"

        if language not in self.label2id:
            label = self.label2id["unknown"]
        else:
            label = self.label2id[language]

        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(pred):
    """Compute evaluation metrics."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)

    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


def finetune_glot500(train_data_path, eval_data_path, output_dir, languages_file_path, num_epochs=3, batch_size=8, freeze_base=False):
    tokenizer = AutoTokenizer.from_pretrained("cis-lmu/glot500-base")

    language_labels = load_language_list(languages_file_path)
    train_dataset = OpenLIDDataset(train_data_path, tokenizer, language_labels)

    label2id = train_dataset.label2id

    eval_dataset = FloresDataset(eval_data_path, tokenizer, label2id)

    model = AutoModelForSequenceClassification.from_pretrained(
        #"cis-lmu/glot500-base",
        f"{output_dir}/checkpoint-400",
        num_labels=len(label2id),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )

    if freeze_base:
        for param in model.base_model.parameters():
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",
        eval_steps=200,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=10,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        optim="adamw_torch",
        learning_rate=5e-5,
        weight_decay=0.1,
        gradient_accumulation_steps=1,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    final_results = trainer.evaluate()
    print(f"Final evaluation results: {final_results}")

    trainer.save_model()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Glot500 for language identification")
    parser.add_argument("--train-data", required=True, help="Path to training data JSONL file")
    parser.add_argument("--eval-data", required=True, help="Path to evaluation data JSONL file")
    parser.add_argument("--output-dir", required=True, help="Output directory for fine-tuned model")
    parser.add_argument("--languages-file", required=True, help="Path to languages.txt file")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size per GPU")
    parser.add_argument("--freeze-base", action="store_true", help="Freeze the base model parameters")

    args = parser.parse_args()

    finetune_glot500(train_data_path=args.train_data, eval_data_path=args.eval_data, output_dir=args.output_dir,
                     languages_file_path=args.languages_file, num_epochs=args.num_epochs, batch_size=args.batch_size,
                     freeze_base=args.freeze_base)
