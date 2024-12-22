import yaml
import os
import json
import argparse
from typing import Optional, Literal, List
import copy
from functools import partial
import re

from tqdm.auto import tqdm
from unsloth import FastLanguageModel
from pydantic import BaseModel
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

from data_utils import load_dataset, LABELS, LANGUAGES
from eval_utils import compute_metrics


def extract_llama_answer(output):
    # Use regex to find the assistant's response
    match = re.search(r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>', output, re.DOTALL)

    if match:
        assistant_answer = match.group(1)
        return assistant_answer
    else:
        return None


SYSTEM = """
You are an expert at detecting emotions in text.
Please classify the text into one of the following categories:
Anger, Fear, Joy, Sadness, Surprise
Your response should be a JSON object with the following format:
{{
    "disgust": bool,
    "anger": bool,
    "fear": bool,
    "joy": bool,
    "sadness": bool,
    "surprise": bool
}}
Do not give explanations. Just return the JSON object.
"""

class Config(BaseModel):
    max_seq_length: int = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype: Optional[str] = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit: bool = True # Use 4bit quantization to reduce memory usage. Can be False.
    fourbit_model: Literal[
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
        "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/gemma-2-9b-bnb-4bit",
        "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
        "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-3B-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct"
    ] = "unsloth/Llama-3.2-3B-Instruct"
    token: str
    r: int = 16
    use_rslora: bool = False
    training_args: dict
    run_name: str
    languages: Optional[List[str]] = None

def get_output_dir(config: Config) -> str:
    path = os.path.join("results", "llm", config.run_name)
    os.makedirs(path, exist_ok=True)
    return path

def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if config["languages"] is None:
        config["languages"] = LANGUAGES
    return Config(**config)

def formatting_prompts_func(examples, tokenizer):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

def convert_to_conversations(example):
    # Ensure 'text' is a string and LABELS are present in the example
    text_value = example.get("text", "")
    label_values = {label: example[label] for label in LABELS if label in example}
    
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM},
            {"from": "human", "value": text_value}, 
            {"from": "gpt", "value": json.dumps(label_values)}
        ],
    }


def load_string_prediction(prediction: Optional[str]):
    if prediction is None:
        return None
    for i in range(len(prediction)):
        try:
            loaded_prediction = json.loads(prediction[:len(prediction)-i])
            if not isinstance(loaded_prediction, dict):
                raise ValueError("Prediction is not a dictionary")
            for key in copy.copy(list(loaded_prediction.keys())):
                if key not in LABELS:
                    del loaded_prediction[key]
            for label in LABELS:
                asigned_value = loaded_prediction.get(label, 0)
                if not isinstance(asigned_value, int):
                    asigned_value = 0
                elif asigned_value > 1:
                    asigned_value = 1
                loaded_prediction[label] = asigned_value
            break
        except (json.JSONDecodeError, ValueError):
            if i == len(prediction) - 1:
                loaded_prediction = None
    return loaded_prediction


def main(
    config_path: str
):
    config = load_config(config_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.fourbit_model, 
        max_seq_length = config.max_seq_length,
        dtype = config.dtype,
        load_in_4bit = config.load_in_4bit,
        token=config.token
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = config.r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = config.use_rslora,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )
    
    formatting_fn = partial(formatting_prompts_func, tokenizer=tokenizer)

    train_dataset = load_dataset(format="datasets", track="a", languages=config.languages)["train"]
    train_dataset = train_dataset.map(convert_to_conversations, remove_columns=train_dataset.column_names)
    val_dataset = load_dataset(format="datasets", track="a", languages=config.languages)["validation"]
    val_ids = val_dataset["id"]
    val_dataset = val_dataset.map(convert_to_conversations, remove_columns=val_dataset.column_names)

    train_dataset = standardize_sharegpt(train_dataset)
    train_dataset = train_dataset.map(formatting_fn, batched = True,)
    val_dataset = standardize_sharegpt(val_dataset)
    val_dataset = val_dataset.map(formatting_fn, batched = True,)


    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text",
        max_seq_length = config.max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            optim = "adamw_8bit",
            seed = 3407,
            output_dir = get_output_dir(config) + "/checkpoints",
            report_to = "none", # Use this for WandB etc
            load_best_model_at_end = True,
            metric_for_best_model="eval_loss",
            eval_strategy = "epoch",
            save_strategy = "epoch",
            **config.training_args
        ),
        eval_dataset = val_dataset,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    trainer.train()


    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    # Assuming val_dataset is a list of conversation dictionaries
    batch_size = 8  # Define your batch size
    batched_answers = []
    predictions_dict = {}
    invalid_predictions = 0

    for i in tqdm(range(0, len(val_dataset), batch_size), desc="Generating predictions for the validation set"):
        batch_messages = [val_dataset[j]["conversations"][:-1] for j in range(i, min(i + batch_size, len(val_dataset)))]
        inputs = tokenizer.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,  # Must add for generation
            return_tensors="pt",
            padding=True,  # Ensure padding for batch processing
            truncation=True,  # Ensure truncation if necessary
        ).to("cuda")

        outputs = model.generate(input_ids=inputs, max_new_tokens=64, use_cache=True,
                                temperature=1.5, min_p=0.1)
        decoded_outputs = tokenizer.batch_decode(outputs)
        answers = [extract_llama_answer(output) for output in decoded_outputs]
        batched_answers.extend(answers)
        
    for j, answer in enumerate(batched_answers):
        val_id = val_ids[j]
        prediction = load_string_prediction(answer)
        if prediction is None:
            invalid_predictions += 1
            predictions_dict[val_id] = {label: 0 for label in LABELS}
        else:
            predictions_dict[val_id] = prediction
    
    print(f"Number of invalid predictions: {invalid_predictions}")
    metrics = compute_metrics(predictions=predictions_dict, split="validation", languages=config.languages)
    
    output_dir = get_output_dir(config)
    metrics_path = os.path.join(output_dir, f"{','.join(config.languages)}_metrics.json")
    predictions_path = os.path.join(output_dir, f"{','.join(config.languages)}_predictions.json")
    model_path = os.path.join(output_dir, "best_checkpoint")
    config_copy_path = os.path.join(output_dir, "config_copy.yaml")
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    with open(predictions_path, "w") as f:
        json.dump(predictions_dict, f, indent=4)

    model.save_pretrained(model_path + "/model")
    tokenizer.save_pretrained(model_path + "/tokenizer")
    
    with open(config_copy_path, "w") as f:
        yaml.dump(config.model_dump(), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
