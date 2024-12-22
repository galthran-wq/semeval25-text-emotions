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
    num_beams: int
    repetition_penalty: float
    no_repeat_ngram_size: int

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
    
    # Modify model generation config to include early stopping, num beams, and repetition penalty
    model.config.early_stopping = True
    model.config.num_beams = config.num_beams
    model.config.repetition_penalty = config.repetition_penalty
    model.config.no_repeat_ngram_size = config.no_repeat_ngram_size
    # Set eos_token_id and pad_token_id if not already set
    if model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id


    # Tokenize data
    def tokenize_function(examples):
        # Convert labels to a comma-separated string of label names where the value is 1
        labels_str = [",".join([label for label in datasets_labels if examples[label][i] == 1]) for i in range(len(examples['text']))]
        # Tokenize the text and labels
        model_inputs = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(labels_str, padding="max_length", truncation=True, max_length=128)
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
        predict_with_generate=True,
        **config.training_params
    )

    def compute_metrics_wrapper(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Convert decoded strings back to a list and compute metrics
        invalid_format_count = 0
        pred_dict = {}
        for val_id, pred in zip(val_ids, decoded_preds):
            pred = pred.replace(" and", ",") # weird format from the model
            try:
                # Split the prediction string into a list of label names
                predicted_labels = [label.strip() for label in pred.split(",") if label.strip() in datasets_labels]
                pred_dict[val_id] = {label: 1 if label in predicted_labels else 0 for label in datasets_labels}
            except Exception as e:
                invalid_format_count += 1
                pred_dict[val_id] = {label: 0 for label in datasets_labels}
        
        print(f"Number of invalid formats: {invalid_format_count}")
        metrics = compute_metrics(predictions=pred_dict, split="validation", languages=config.languages)
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
