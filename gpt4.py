import os
import json
from typing import Optional
import pandas as pd

from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
import fire

from data_utils import load_dataset, LABELS
from eval_utils import compute_metrics
from langchain.retrievers import Retriever
from langchain.embeddings import BGEEmbedding
from langchain_core.runnables import MMR

openai = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

class ClassificationResult(BaseModel):
    disgust: Optional[bool] = False
    anger: Optional[bool] = False
    fear: Optional[bool] = False
    joy: Optional[bool] = False
    sadness: Optional[bool] = False
    surprise: Optional[bool] = False


def get_zeroshot_chain():
    system = """
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
    prompt =ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", "{text}")
    ])
    chain = (
        {"text": RunnablePassthrough()} |
        prompt |
        openai |
        PydanticOutputParser(pydantic_object=ClassificationResult)
    )
    return chain


def load_data_for_language(track: str, language: str, data_root: str = "./public_data", split: str = "validation") -> pd.DataFrame:
    data = load_dataset(track=track, data_root=data_root, format="pandas")
    data = data[split]
    data = data[data["language"] == language]
    return data


def get_retriever_with_examples(
    track: str,
    language: str,
    data_root: str = "./public_data",
    num_examples: int = 5,
    use_mmr: bool = False
) -> Retriever:
    # Load the training data
    train_data = load_data_for_language(track, language, data_root, split="train")

    # Initialize the BGE embedding model
    embedding_model = BGEEmbedding(model_name="BAAI/bge-m3")

    # Create a retriever
    retriever = Retriever(
        embedding_model=embedding_model,
        documents=train_data["text"].tolist(),
        ids=train_data["id"].tolist()
    )

    # Optionally apply MMR
    if use_mmr:
        retriever = MMR(retriever)

    # Retrieve examples
    def retrieve_examples(query: str):
        return retriever.retrieve(query, num_examples=num_examples)

    return retrieve_examples


def main(
    track: str = "a",
    language: str = "eng",
    data_root: str = "./public_data",
    predictions_dir: str = "results/gpt/zeroshot",
    num_workers: int = 7
):
    import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed

    os.makedirs(predictions_dir, exist_ok=True)
    predictions_file = os.path.join(predictions_dir, f"{language}_predictions.json")
    
    # Load existing predictions if they exist
    if os.path.exists(predictions_file):
        with open(predictions_file, "r") as f:
            predictions = json.load(f)
    else:
        predictions = {}

    data = load_data_for_language(track, language, data_root)
    texts_to_predict = []
    ids_to_predict = []

    # Determine which entries need predictions
    for id, text in zip(data["id"], data["text"]):
        if id not in predictions:
            texts_to_predict.append(text)
            ids_to_predict.append(id)

    # Run the chain only for entries that need predictions
    if texts_to_predict:
        chain = get_zeroshot_chain()
        results = {}

        def process_text(id, text):
            try:
                result = chain.invoke({"text": text})
                return id, result
            except Exception as e:
                print(f"Error processing text ID {id}: {e}")
                return id, None

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_id = {executor.submit(process_text, id, text): id for id, text in zip(ids_to_predict, texts_to_predict)}
            for future in tqdm.tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Processing texts"):
                id, result = future.result()
                if result is not None:
                    results[id] = {label: result.model_dump()[label] for label in LABELS}

        predictions.update(results)

        # Save updated predictions
        with open(predictions_file, "w") as f:
            json.dump(predictions, f)
    
    metrics = compute_metrics(predictions, languages=[language])
    with open(os.path.join(predictions_dir, f"{language}_metrics.json"), "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    fire.Fire(main)
