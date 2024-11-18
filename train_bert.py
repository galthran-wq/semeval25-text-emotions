import copy
import os
import json
import argparse
from typing import List, Optional
from datetime import datetime

import datasets
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import yaml

from eval_utils import compute_metrics
from data_utils import load_dataset, LABELS

class Config(BaseModel):
    track: str
    languages: List[str]
    checkpoint_path: str
    training_params: dict
    run_name: str

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate BERT model.")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the YAML configuration file.')
    return parser.parse_args()

def get_output_dir(config: Config) -> str:
    path = os.path.join("results", "bert", config.run_name)
    os.makedirs(path, exist_ok=True)
    return path

def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)

def train_and_evaluate(config: Config):
    # Load dataset
    track_datasets = load_dataset(track=config.track, format="pandas")
    train_data = track_datasets["train"]
    val_data = track_datasets["validation"]

    # Filter by languages
    train_data = train_data[train_data['language'].isin(config.languages)]
    val_data = val_data[val_data['language'].isin(config.languages)]
    val_ids = val_data["id"].tolist()

    datasets_labels = copy.deepcopy(LABELS)
    for label in datasets_labels:
        if val_data[label].isna().any():
            datasets_labels.remove(label)

    train_data = train_data[["text"] + datasets_labels]
    val_data = val_data[["text"] + datasets_labels]
    print(f"Only keep {len(datasets_labels)} labels")

    train_data = datasets.Dataset.from_pandas(train_data)
    val_data = datasets.Dataset.from_pandas(val_data)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.checkpoint_path, 
        num_labels=len(datasets_labels),
        problem_type="multi_label_classification"
    )

    # Tokenize data
    def tokenize_function(examples):
        # Tokenize the text and return input_ids, attention_mask, and labels
        tokenized_inputs = tokenizer(examples['text'], padding="max_length", truncation=True)
        # Ensure labels are included in the tokenized output
        # Convert labels to float
        tokenized_inputs['labels'] = [list(map(float, label_set)) for label_set in zip(*[examples[label] for label in datasets_labels])]
        return tokenized_inputs

    train_data = train_data.map(tokenize_function, batched=True)
    val_data = val_data.map(tokenize_function, batched=True)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=get_output_dir(config) + f"/checkpoints",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_strategy="no",
        **config.training_params
    )

    def compute_metrics_wrapper(eval_pred):
        predictions, labels = eval_pred
        predictions = torch.sigmoid(torch.tensor(predictions)) > 0
        pred_dict = {
            val_ids[prediction_index]: {
                label: int(predictions[prediction_index][i].item())
                for i, label in enumerate(datasets_labels)
            } 
            for prediction_index in range(len(predictions))
        }
        return compute_metrics(pred_dict, split="validation", languages=config.languages)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics_wrapper
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(os.path.join(get_output_dir(config), f"{','.join(config.languages)}_best_model"))

    # Evaluate the model
    metrics = trainer.evaluate()
    metrics_path = os.path.join(get_output_dir(config), f"{','.join(config.languages)}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config_path)
    train_and_evaluate(config)
