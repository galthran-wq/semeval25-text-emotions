import os
from datetime import datetime
from itertools import chain, combinations
from tqdm import tqdm
from typing import Literal, List
import yaml
import json
import random

import fire
import pandas as pd
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

from data_utils import load_dataset, LABELS


class Config(BaseModel):
    track: str
    language: str
    model: str
    num_workers: int
    num_iterations: int
    num_fewshot_examples: int
    batch_size: int

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return cls(**config_dict)


def get_output_dir(config: Config) -> str:
    path = os.path.join("results", "synth_data", config.language, f"{config.model}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(path, exist_ok=True)
    return path

def batchify(iterable, batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i+batch_size]

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def sample_labels_combination(train_data: pd.DataFrame) -> List[str]:
    existing_labels = [label for label in LABELS if label in train_data.columns and train_data[label].isna().sum() == 0]
    print(f"Existing labels: {existing_labels}")
    all_possible_labels_combinations = list(powerset(existing_labels))
    print(f"The number of all possible labels combinations is {len(all_possible_labels_combinations)}")
    existing_labels_combinations = []
    weights = []
    for combination in all_possible_labels_combinations:
        not_combination = [label for label in existing_labels if label not in combination]
        if len(combination) > 0:
            combination_mask = ((train_data[list(combination)] == 1).all(axis=1) & (train_data[list(not_combination)] == 0).all(axis=1))
        else:
            combination_mask = (train_data[list(not_combination)] == 0).all(axis=1)
        if combination_mask.any():
            combination_weight = combination_mask.sum()
            existing_labels_combinations.append(combination)
            weights.append(combination_weight)
    assert sum(weights) == len(train_data), f"The sum of weights should be equal to the number of rows in the train data, but len(train_data)={len(train_data)} and sum(weights)={sum(weights)}"
    for combination, combination_weight in sorted(zip(existing_labels_combinations, weights), key=lambda x: x[1], reverse=True):
        print(f"Combination: {combination}, Weight: {combination_weight}")
    return list(random.choices(existing_labels_combinations, weights=weights, k=1)[0])


def generate_prompt(clarity: Literal["low", "medium", "high"], num_words: int, examples: List[str], emotions: List[str]):
    prompt = f"""
You are an expert at detecting emotions in text. Your task is to generate text examples with a provided set of emotions.

Generate a text with clarity level {clarity} and approximately {num_words} words. 

The emotions that should be present in the text are: {", ".join(emotions) or "none"}.

Your response should be a text that has the specified emotions. Do not explain yourself or the task.

Here are some examples of such texts:
"""
    for example in examples:
        prompt += f"{example}\n"
    prompt += "\nNow generate a new example:"
    return prompt


def generate_prompts(train_data: pd.DataFrame, num_samples: int = 5, n_fewshot_examples: int = 10):
    prompts = []
    for _ in range(num_samples):
        emotions = sample_labels_combination(train_data)
        emotions_data = train_data[(train_data[emotions] == 1).any(axis=1)]
        clarity = random.choice(['low', 'medium', 'high'])
        num_words = random.randint(5, 20)
        examples = emotions_data.sample(n=n_fewshot_examples, replace=False)["text"].tolist()
        prompt = generate_prompt(clarity, num_words, examples)
        prompts.append({
            "emotions": emotions,
            "clarity": clarity,
            "num_words": num_words,
            "examples": examples,
            "prompt": prompt,
        })
    return prompts

def setup_llm(model: str):
    from langchain_ollama.llms import OllamaLLM
    return OllamaLLM(model=model)

def generate_responses(llm: LLM, prompts: List[str], num_workers: int = 2):
    chain = llm | StrOutputParser()
    responses: List[str] = chain.batch(prompts, config={"max_workers": num_workers})
    return responses


def main(config_path: str):
    config = Config.from_yaml(config_path)
    output_dir = get_output_dir(config)
    llm = setup_llm(config.model)
    train_data = load_dataset(track=config.track, format="pandas", languages=[config.language])["train"]
    prompts = generate_prompts(train_data, num_samples=config.num_iterations, n_fewshot_examples=config.num_fewshot_examples)
    for i, batch in tqdm(enumerate(batchify(prompts, config.batch_size)), desc="Generating responses..."):
        batch_responses = generate_responses(llm, batch, num_workers=config.num_workers)
        batch_responses_path = os.path.join(output_dir, f"batch_{i}.json")
        print(f"Generated {len(batch_responses)} responses for batch {i}... Saving to {batch_responses_path}")
        with open(batch_responses_path, "w") as file:
            results = []
            for prompt, response in zip(batch, batch_responses):
                del prompt["prompt"]
                results.append({"prompt": prompt, "response": response})
            json.dump(results, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    fire.Fire(main)
