import copy
import os
import json
import argparse
from typing import List, Optional
from datetime import datetime

import datasets
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
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
    parser = argparse.ArgumentParser(description="Train and evaluate Seq2Seq model.")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the YAML configuration file.')
    return parser.parse_args()

def get_output_dir(config: Config) -> str:
    path = os.path.join("results", "seq2seq", config.run_name)
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
    model = AutoModelForSeq2SeqLM.from_pretrained(config.checkpoint_path)

    # Tokenize data
    def tokenize_function(examples):
        # Convert labels to JSON string
        labels_json = [json.dumps({label: int(examples[label][i]) for label in datasets_labels}) for i in range(len(examples['text']))]
        # Tokenize the text and labels
        model_inputs = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(labels_json, padding="max_length", truncation=True, max_length=128)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    train_data = train_data.map(tokenize_function, batched=True)
    val_data = val_data.map(tokenize_function, batched=True)

    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=get_output_dir(config) + f"/checkpoints",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        **config.training_params
    )

    def compute_metrics_wrapper(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Convert decoded strings back to JSON and compute metrics
        invalid_json_count = 0
        pred_dict = {}
        for val_id, pred in zip(val_ids, decoded_preds):
            try:
                pred_dict[val_id] = json.loads(pred)
            except json.JSONDecodeError:
                invalid_json_count += 1
                pred_dict[val_id] = {label: 0 for label in datasets_labels}
        
        label_dict = {val_id: json.loads(label) for val_id, label in zip(val_ids, decoded_labels)}
        
        print(f"Number of invalid JSONs: {invalid_json_count}")
        
        metrics = compute_metrics(pred_dict, label_dict, split="validation", languages=config.languages)
        return metrics

    trainer = Seq2SeqTrainer(
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
