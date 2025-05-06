import json
from typing import Literal, Union, Dict
from pathlib import Path
import datasets
import pandas as pd
from sklearn.model_selection import train_test_split

import pandas as pd

LABELS = ["disgust", "anger", "fear", "joy", "sadness", "surprise"]
LANGUAGES = [
    "eng", 
    "amh", 
    "deu", 
    "ptbr", 
    "rus", 
    "som", 
    "sun", 
    "tir",
    "afr",
    "arq",
    "ary",
    "chn",
    "esp",
    "hau",
    "hin",
    "ibo",
    "kin",
    "mar",
    "orm",
    "pcm",
    "ptmz",
    "ron",
    "swa",
    "swe",
    "tat",
    "ukr",
    "vmw",
    "yor",
]
LOW_RESOURCE_LANGUAGES = [
    "afr",
    "amh",
    "vmw",
    "hau",
    "ibo",
    "xho",
    "zul",
    "jav",
    "kin",
    "mar",
    "pcm",
    "orm",
    "som",
    "swa",
    "sun",
    "tir",
    "yor",
    "tat"
]

def save_data_split(
    track: Literal["a", "b"], 
    data_root="./public_data", 
) -> Union[Dict[str, pd.DataFrame], datasets.DatasetDict]:
    data_root = Path(data_root)
    train_data_root = data_root / "train" / f"track_{track}"

    # Collect ids by language
    language_ids = {}
    for language_data_file in train_data_root.glob("*.csv"):
        language = language_data_file.stem
        data = pd.read_csv(language_data_file)
        language_ids[language] = data["id"].tolist()
        
    # Check for duplicates
    all_ids = [id for ids in language_ids.values() for id in ids]
    assert len(all_ids) == len(set(all_ids)), "Duplicate ids"

    # Split each language's data with same ratio
    train_ids = []
    test_ids = []
    for language, ids in language_ids.items():
        lang_train_ids, lang_test_ids = train_test_split(
            ids, test_size=0.2, random_state=42
        )
        train_ids.extend(lang_train_ids)
        test_ids.extend(lang_test_ids)

    split_file = data_root / "split_ids.json"
    if split_file.exists():
        with open(split_file, "r") as f:
            split_data = json.load(f)
        assert set(split_data["train_ids"]) == set(train_ids), "Train IDs do not match"
        assert set(split_data["val_ids"]) == set(test_ids), "Validation IDs do not match"
    else:
        split_data = {"train_ids": train_ids, "val_ids": test_ids}
        with open(split_file, "w") as f:
            json.dump(split_data, f)

    return data


def load_dataset(
    track: Literal["a", "b"], 
    data_root="./public_data", 
    format: Literal["pandas", "datasets"] = "pandas", 
    languages=None
) -> Union[Dict[str, pd.DataFrame], datasets.DatasetDict]:
    data_root = Path(data_root)
    train_data_root = data_root / f"track_{track}" / "train"
    dev_data_root = data_root / f"track_{track}" / "dev"
    test_data_root = data_root / f"track_{track}" / "test"

    # Load split IDs
    split_file = data_root / "split_ids.json"
    with open(split_file, "r") as f:
        split_data = json.load(f)
    train_ids = set(split_data["train_ids"])
    val_ids = set(split_data["val_ids"])

    # Load training data
    train_data = []
    val_data = []
    for language_data_file in train_data_root.glob("*.csv"):
        language = language_data_file.stem
        if languages is not None and language not in languages:
            continue
        data = pd.read_csv(language_data_file)
        data.columns = data.columns.str.lower()
        data["language"] = language
        if track == "a":
            for col in LABELS:
                if col not in data.columns:
                    data[col] = None
                    
        # Split into train and validation based on IDs
        train_mask = data["id"].isin(train_ids)
        train_data.append(data[train_mask])
        val_data.append(data[~train_mask])
        
    train_data = pd.concat(train_data, axis=0).reset_index(drop=True)
    val_data = pd.concat(val_data, axis=0).reset_index(drop=True)

    # Load dev data
    dev_data = []
    for language_data_file in dev_data_root.glob("*.csv"):
        language = language_data_file.stem.split("_")[0]
        if languages is not None and language not in languages:
            continue
        data = pd.read_csv(language_data_file)
        data.columns = data.columns.str.lower()
        data["language"] = language
        if track == "a":
            for col in LABELS:
                if col not in data.columns:
                    data[col] = None
        dev_data.append(data)
    dev_data = pd.concat(dev_data, axis=0).reset_index(drop=True)

    # Load test data
    test_data = []
    for language_data_file in test_data_root.glob("*.csv"):
        language = language_data_file.stem.split("_")[0]
        if languages is not None and language not in languages:
            continue
        data = pd.read_csv(language_data_file)
        data.columns = data.columns.str.lower()
        data["language"] = language
        test_data.append(data)
    test_data = pd.concat(test_data, axis=0).reset_index(drop=True)

    train_full_data = pd.concat([train_data, val_data])
    train_full_with_dev_data = pd.concat([train_data, val_data, dev_data])

    if format == "datasets":
        train_data = datasets.Dataset.from_pandas(train_data)
        val_data = datasets.Dataset.from_pandas(val_data)
        dev_data = datasets.Dataset.from_pandas(dev_data)
        train_full_data = datasets.Dataset.from_pandas(train_full_data)
        train_full_with_dev_data = datasets.Dataset.from_pandas(train_full_with_dev_data)

    data = {
        "train": train_data, 
        "validation": val_data, 
        "dev": dev_data,
        "test": test_data,
        "train_full": train_full_data,
        "train_full_with_dev": train_full_with_dev_data,
    }
    if format == "datasets":
        data = datasets.DatasetDict(data)
    return data


def load_data_for_language( language: str, track: str = "a", data_root: str = "./public_data", split: Literal["train", "validation", "dev", "train_full", "train_full_with_dev"] = "validation") -> pd.DataFrame:
    data = load_dataset(track=track, data_root=data_root, format="pandas")
    data = data[split]
    data = data[data["language"] == language]
    return data
