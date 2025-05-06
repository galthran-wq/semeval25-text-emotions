from typing import Dict, Literal, List
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data_utils import load_dataset

LABELS = ["disgust", "anger", "fear", "joy", "sadness", "surprise"]

def _get_language_from_id(id: str) -> str:
    return id.split("_")[0]


def compute_metrics(
    predictions: Dict[str, Dict[str, int]], 
    split: Literal["validation", "train", "dev"] = "validation",
    languages: List[str] = None
) -> Dict[str, float]:
    """
    predictions: { $id: { $label: $prediction } }
    """
    dataset = load_dataset(track="a", format="pandas")[split]
    # { $id: { $label: $prediction } }
    dataset_labels_entries = dataset[["id", *LABELS]].set_index("id").to_dict(orient="index")
    if languages is not None:
        dataset_labels_entries = {
            id: entries for id, entries in dataset_labels_entries.items() 
            if _get_language_from_id(id) in languages
        }
    all_languages = list(set([_get_language_from_id(id) for id in dataset_labels_entries.keys()]))
    
    # Validate predictions, assert that all ids are the same
    missing_ids = list(set(dataset_labels_entries.keys()).difference(set(predictions.keys())))
    print(f"Missing {len(missing_ids)} ids! Will set all the predictions to zero!")
    for id in missing_ids:
        predictions[id] = {label: 0 for label in LABELS}

    # Calculate metrics by label
    # Calculate metrics by language
    by_language = {}
    for language in all_languages:
        lang_y_true = []
        lang_y_pred = []
        for id, entries in dataset_labels_entries.items():
            if _get_language_from_id(id) == language:
                true_labels = [entries[label] for label in LABELS if label in entries and entries[label] is not None]
                pred_labels = [predictions[id][label] for label in LABELS if label in entries and entries[label] is not None]
                lang_y_true.append(true_labels)
                lang_y_pred.append(pred_labels)
        lang_y_true = np.array(lang_y_true)
        lang_y_pred = np.array(lang_y_pred)
        by_language[language] = {
            "accuracy": round(accuracy_score(lang_y_true, lang_y_pred), 4),
            "f1_macro": round(f1_score(lang_y_true, lang_y_pred, average='macro'), 4),
            "precision_macro": round(precision_score(lang_y_true, lang_y_pred, average='macro'), 4),
            "recall_macro": round(recall_score(lang_y_true, lang_y_pred, average='macro'), 4),
            "f1_micro": round(f1_score(lang_y_true, lang_y_pred, average='micro'), 4),
            "precision_micro": round(precision_score(lang_y_true, lang_y_pred, average='micro'), 4),
            "recall_micro": round(recall_score(lang_y_true, lang_y_pred, average='micro'), 4),
            "n": len(lang_y_true)
        }

    # Compute macro and micro F1 from by_language
    f1_macro_macro = round(np.mean([metrics["f1_macro"] for metrics in by_language.values()]), 4)
    f1_micro_macro = round(np.average([metrics["f1_micro"] for metrics in by_language.values()], weights=[metrics["n"] for metrics in by_language.values()]), 4)
    f1_macro_micro = round(np.mean([metrics["f1_macro"] for metrics in by_language.values()]), 4)
    f1_micro_micro = round(np.average([metrics["f1_micro"] for metrics in by_language.values()], weights=[metrics["n"] for metrics in by_language.values()]), 4)

    # Calculate metrics by label
    by_label = {}
    for label in LABELS:
        label_y_true = []
        label_y_pred = []
        for id, entries in dataset_labels_entries.items():
            if label in entries and entries[label] is not None:
                true_labels = entries[label]
                pred_labels = predictions[id][label]
                label_y_true.append(true_labels)
                label_y_pred.append(pred_labels)
        by_label[label] = {
            "accuracy": round(accuracy_score(label_y_true, label_y_pred), 4),
            "f1": round(f1_score(label_y_true, label_y_pred), 4),
            "precision": round(precision_score(label_y_true, label_y_pred), 4),
            "recall": round(recall_score(label_y_true, label_y_pred), 4),
            "n": len(label_y_true)
        }

    # Calculate metrics by language and label
    by_language_by_label = {}
    for language in all_languages:
        by_language_by_label[language] = {}
        for label in LABELS:
            lang_label_y_true = []
            lang_label_y_pred = []
            for id, entries in dataset_labels_entries.items():
                if _get_language_from_id(id) == language and label in entries and entries[label] is not None:
                    lang_label_y_true.append(entries[label])
                    lang_label_y_pred.append(predictions[id][label])

            by_language_by_label[language][label] = {
                "accuracy": round(accuracy_score(lang_label_y_true, lang_label_y_pred), 4),
                "f1": round(f1_score(lang_label_y_true, lang_label_y_pred), 4),
                "precision": round(precision_score(lang_label_y_true, lang_label_y_pred), 4),
                "recall": round(recall_score(lang_label_y_true, lang_label_y_pred), 4),
            }

    return {
        "f1_macro_macro": f1_macro_macro,
        "f1_micro_macro": f1_micro_macro,
        "f1_macro_micro": f1_macro_micro,
        "f1_micro_micro": f1_micro_micro,
        "by_language": by_language,
        "by_label": by_label,
        "by_language_by_label": by_language_by_label
    }
