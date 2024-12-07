import os
import json
from typing import Optional, Literal
import pandas as pd
from typing import List
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
import fire

from data_utils import load_dataset, LABELS
from eval_utils import compute_metrics
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore

openai = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")


class RetrieverExampleSelector(BaseExampleSelector):
    def __init__(self, retriever):
        self.retriever = retriever

    def add_example(self, example):
        raise NotImplementedError
    
    def format_docs(self, docs):
        return [
            {
                "text": doc.page_content, 
                "result": doc.metadata["result"]
            }
            for doc in docs
        ]

    def select_examples(self, input_variables):
        # This assumes knowledge that part of the input will be a 'text' key
        new_text = input_variables["text"]
        best_k_docs = self.retriever.invoke(new_text)
        return self.format_docs(best_k_docs)


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


def load_data_for_language( language: str, track: str = "a", data_root: str = "./public_data", split: Literal["train", "validation", "dev", "train_full"] = "validation") -> pd.DataFrame:
    data = load_dataset(track=track, data_root=data_root, format="pandas")
    if split == "train_full":
        data = pd.concat([data["train"], data["validation"]])
    else:
        data = data[split]
    data = data[data["language"] == language]
    return data


def get_bge_retriever(data: pd.DataFrame, model_name: str, device: str = "cpu", n_examples: int = 5, mrr: bool = False):
    data: List[Document] = [
        Document(
            page_content=entry["text"], 
            metadata={"result": ClassificationResult(
                anger=entry["anger"],
                fear=entry["fear"],
                joy=entry["joy"],
                sadness=entry["sadness"],
                surprise=entry["surprise"]
            ).model_dump_json()}
        )
        for i, entry in data.iterrows()
    ]
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    if model_name == "BAAI/bge-m3":
        hf = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs, query_instruction=""
        )
    else:
        hf = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
    db = SKLearnVectorStore.from_documents(data, hf)
    retriever = db.as_retriever(search_kwargs={"k": n_examples}, search_type="similarity" if not mrr else "mmr")
    return retriever


def get_fewshot_chain(retriever):
    system = """
You are an expert at detecting emotions in text.
You are given examples of texts and their corresponding emotions.
Please classify the new text into one of the following categories:
Anger, Fear, Joy, Sadness, Surprise
Your response should be a JSON object with the following format:
{{
    "anger": bool,
    "fear": bool,
    "joy": bool,
    "sadness": bool,
    "surprise": bool
}}
Do not give explanations. Just return the JSON object.
"""
    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "{text}"),
        ("assistant", "{result}"),
    ])
    example_selector = RetrieverExampleSelector(retriever)
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        example_selector=example_selector
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        few_shot_prompt,
        ("user", "{text}"),
    ])
    chain = (
        {"text": RunnablePassthrough()} |
        prompt |
        openai |
        PydanticOutputParser(pydantic_object=ClassificationResult)
    )
    return chain


def fewshot(
    *,
    language: str = "eng",
    data_root: str = "./public_data",
    predictions_dir: str = "results/gpt/fewshot",
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
    num_examples: int = 100,
    use_mmr: bool = False,
    num_workers: int = 7
):
    # Load the training data
    train_data = load_data_for_language(language=language, data_root=data_root, split="train")

    # Initialize the BGE embedding model
    retriever = get_bge_retriever(train_data, model_name, device, num_examples, use_mmr)

    chain = get_fewshot_chain(retriever)

    os.makedirs(predictions_dir, exist_ok=True)
    predictions_file = os.path.join(predictions_dir, f"{language}_{model_name.split('/')[-1]}_{num_examples}shot_{'mmr' if use_mmr else 'cosine'}_predictions.json")
    
    # Load existing predictions if they exist
    if os.path.exists(predictions_file):
        with open(predictions_file, "r") as f:
            predictions = json.load(f)
    else:
        predictions = {}

    data = load_data_for_language(language=language, data_root=data_root, split="validation")
    texts_to_predict = []
    ids_to_predict = []

    # Determine which entries need predictions
    for id, text in zip(data["id"], data["text"]):
        if id not in predictions:
            texts_to_predict.append(text)
            ids_to_predict.append(id)

    # Run the chain only for entries that need predictions
    if texts_to_predict:
        chain = get_fewshot_chain(retriever)
        results = run_chain(chain=chain, texts=texts_to_predict, ids=ids_to_predict, num_workers=num_workers)
        predictions.update(results)

        # Save updated predictions
        with open(predictions_file, "w") as f:
            json.dump(predictions, f)
    
    metrics = compute_metrics(predictions, languages=[language])
    with open(os.path.join(predictions_dir, f"{language}_{model_name.split('/')[-1]}_{num_examples}shot_{'mmr' if use_mmr else 'cosine'}_metrics.json"), "w") as f:
        json.dump(metrics, f)

    return chain

def run_chain(chain, texts, ids, num_workers: int = 7):
    results = {}

    def process_text(id, text):
        try:
            result = chain.invoke(text)
            return id, result
        except Exception as e:
            print(f"Error processing text ID {id}: {e}")
            return id, None

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_id = {executor.submit(process_text, id, text): id for id, text in zip(ids, texts)}
        for future in tqdm.tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Processing texts"):
            id, result = future.result()
            if result is not None:
                results[id] = {label: result.model_dump()[label] for label in LABELS}
    return results

def zeroshot(
    track: str = "a",
    language: str = "eng",
    data_root: str = "./public_data",
    predictions_dir: str = "results/gpt/zeroshot",
    num_workers: int = 7
):
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
        results = run_chain(chain, texts_to_predict, ids_to_predict, num_workers)
        predictions.update(results)

        # Save updated predictions
        with open(predictions_file, "w") as f:
            json.dump(predictions, f)
    
    metrics = compute_metrics(predictions, languages=[language])
    with open(os.path.join(predictions_dir, f"{language}_metrics.json"), "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    fire.Fire()
