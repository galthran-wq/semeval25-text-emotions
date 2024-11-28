import copy
import os
import json
import argparse
from typing import List, Optional, Literal
from datetime import datetime

import datasets
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, DebertaV2Tokenizer
import yaml
from sklearn.metrics import roc_auc_score

from eval_utils import compute_metrics
from data_utils import load_dataset, LABELS
from setfit import SetFitModel, TrainingArguments, Trainer


class Config(BaseModel):
    track: str
    languages: List[str]
    checkpoint_path: str
    training_params: dict
    run_name: str
    multi_target_strategy: Literal["multi-output", "classifier-chain", "one-vs-rest"]
    use_differentiable_head: bool

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate SetFit model.")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the YAML configuration file.')
    return parser.parse_args()

def get_output_dir(config: Config) -> str:
    path = os.path.join("results", "setfit", config.run_name)
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

    def map_labels(row):
        row["label"] = [row[label] for label in datasets_labels]
        return row

    train_data = train_data.apply(map_labels, axis=1)
    val_data = val_data.apply(map_labels, axis=1)
    train_data = train_data[["id", "text", "label"]]
    val_data = val_data[["id", "text", "label"]]

    print(f"Only keep {len(datasets_labels)} labels")

    train_data = datasets.Dataset.from_pandas(train_data)
    val_data = datasets.Dataset.from_pandas(val_data)

    # Load tokenizer and model
    head_params = {"num_labels": len(datasets_labels)} if config.use_differentiable_head else None
    model = SetFitModel.from_pretrained(
        config.checkpoint_path,
        multi_target_strategy=config.multi_target_strategy,
        use_differentiable_head=config.use_differentiable_head,
        head_params=head_params
    )
    model.labels = datasets_labels

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=get_output_dir(config) + f"/checkpoints",
        metric_for_best_model="eval_embedding_loss",
        save_strategy="steps",
        save_steps=5000,
        evaluation_strategy="steps",
        eval_steps=5000,
        load_best_model_at_end=True,
        **config.training_params
    )

    def compute_metrics_wrapper(eval_pred):
        binary_predictions, labels = eval_pred
        
        # Calculate AUC-ROC for each label
        # auc_roc = roc_auc_score(labels, probabilities, average='macro')
        
        pred_dict = {
            val_ids[prediction_index]: {
                label: int(binary_predictions[prediction_index][i].item())
                for i, label in enumerate(datasets_labels)
            } 
            for prediction_index in range(len(binary_predictions))
        }
        
        # Compute other metrics
        metrics = compute_metrics(pred_dict, split="validation", languages=config.languages)
        
        return metrics

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    model = trainer.model
    predictions = model.predict(val_data["text"], as_numpy=True, use_labels=False)
    metrics = compute_metrics_wrapper((predictions, None))

    metrics_path = os.path.join(get_output_dir(config), f"{','.join(config.languages)}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config_path)
    train_and_evaluate(config)
