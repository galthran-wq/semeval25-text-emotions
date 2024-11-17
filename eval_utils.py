from typing import Dict, Literal

from sklearn.metrics import accuracy_score, f1_score

from data_utils import load_dataset

LABELS = ["disgust", "anger", "fear", "joy", "sadness", "surprise"]

def _get_language_from_id(id: str) -> str:
    return id.split("_")[0]

def compute_metrics(predictions: Dict[str, Dict[str, int]], labels: Dict[str, Dict[str, float]] = None, split: Literal["validation", "train"] = "validation") -> Dict[str, float]:
    """
    predictions: { $id: { $label: $prediction } }
    """
    dataset = load_dataset(track="a", format="pandas")[split]
    # { $id: { $label: $prediction } }
    dataset_labels_entries = dataset[["id", *LABELS]].set_index("id").to_dict(orient="index")
    all_languages = list(set([_get_language_from_id(id) for id in dataset_labels_entries.keys()]))
    
    # Validate predictions, assert that all ids are the same
    assert set(predictions.keys()) == set(dataset_labels_entries.keys())

    # Calculate overall metrics
    y_true = []
    y_pred = []
    for id in predictions.keys():
        for label in LABELS:
            # not all the data has all the labels
            if label in dataset_labels_entries[id] and dataset_labels_entries[id][label] is not None:
                y_true.append(dataset_labels_entries[id][label])
                y_pred.append(predictions[id][label])

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    # Calculate metrics by language
    by_language = {}
    for language in all_languages:
        lang_y_true = []
        lang_y_pred = []
        for id, entries in dataset_labels_entries.items():
            if _get_language_from_id(id) == language:
                for label in LABELS:
                    if label in entries and entries[label] is not None:
                        lang_y_true.append(entries[label])
                        lang_y_pred.append(predictions[id][label])
        by_language[language] = {
            "accuracy": accuracy_score(lang_y_true, lang_y_pred),
            "f1_macro": f1_score(lang_y_true, lang_y_pred, average='macro'),
            "f1_micro": f1_score(lang_y_true, lang_y_pred, average='micro')
        }

    # Calculate metrics by label
    by_label = {}
    for label in LABELS:
        label_y_true = [
            dataset_labels_entries[id][label] 
            for id in predictions.keys() 
            if label in dataset_labels_entries[id] and dataset_labels_entries[id][label] is not None
        ]
        label_y_pred = [
            predictions[id][label] 
            for id in predictions.keys() 
            if label in dataset_labels_entries[id] and dataset_labels_entries[id][label] is not None
        ]
        by_label[label] = {
            "accuracy": accuracy_score(label_y_true, label_y_pred),
            "f1_macro": f1_score(label_y_true, label_y_pred, average='macro'),
            "f1_micro": f1_score(label_y_true, label_y_pred, average='micro')
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
                "accuracy": accuracy_score(lang_label_y_true, lang_label_y_pred),
                "f1_macro": f1_score(lang_label_y_true, lang_label_y_pred, average='macro'),
                "f1_micro": f1_score(lang_label_y_true, lang_label_y_pred, average='micro')
            }

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "by_language": by_language,
        "by_label": by_label,
        "by_language_by_label": by_language_by_label
    }
