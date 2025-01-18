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
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
import fire

from data_utils import load_dataset, LABELS, LANGUAGES
from eval_utils import compute_metrics
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore

def setup_llm(model: str):
    if model == "gpt-4o":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
    else:
        from langchain_ollama import OllamaLLM
        llm = OllamaLLM(model=model)
    return llm


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


def get_zeroshot_chain(llm):
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
        llm |
        PydanticOutputParser(pydantic_object=ClassificationResult)
    )
    return chain


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


def get_fewshot_chain(retriever, llm, language):
    language_map = {
        "rus": "Russian",
        "eng": "English",
        "afr": "Afrikaans",
        "amh": "Amharic",
        "ptbr": "Portuguese",
        "zho": "Chinese",
        "vmw": "Emakhuwa",
        "eng": "English",
        "deu": "German",
        "hau": "Hausa",
        "hin": "Hindi",
        "ibo": "Igbo",
        "ind": "Indonesian",
        "xho": "isiXhosa",
        "zul": "isiZulu",
        "jav": "Javanese",
        "kin": "Kinyarwanda",
        "esp": "Spanish",
        "mar": "Marathi",
        "rabic": "Moroccan",
        "Pidgin": "Nigerian",
        "orm": "Oromo",
        "ron": "Romanian",
        "rus": "Russian",
        "som": "Somali",
        "sun": "Sundanese",
        "swa": "Swahili",
        "swe": "Swedish",
        "tat": "Tatar",
        "tir": "Tigrinya",
        "ukr": "Ukrainian",
        "yor": "Yoruba"
    }
    if False:
        system = f"Ты -- эксперт по распознованию эмоций в тексте."
        system += """
Пожалуйста, классифицируй предоставленное предожение в несколько категорий, в зависимости от эмоций, представленных в тексте:
Anger --  злость,
Fear -- страх,
Joy -- радость,
Sadness -- грусть,
Surprise -- неожиданность,
Disgust -- отвращение.

Твой ответ обязан соответствовать следующему формату JSON:
{{
    "anger": bool,
    "fear": bool,
    "joy": bool,
    "sadness": bool,
    "surprise": bool,
    "disgust": bool
}}
Не давай пояснения своему ответу. Просто верни JSON объект.
"""
    else:
        system = f"You are an expert at detecting emotions in text. The texts are given in {language_map[language]} language."
        system += """
Please classify the text into one of the following categories:
Anger, Fear, Joy, Sadness, Surprise, Disgust
Your response should be a JSON object with the following format:
{{
    "anger": bool,
    "fear": bool,
    "joy": bool,
    "sadness": bool,
    "surprise": bool,
    "disgust": bool
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
        llm |
        PydanticOutputParser(pydantic_object=ClassificationResult)
    )
    return chain


def fewshot(
    *,
    model: str,
    language: str = "eng",
    data_root: str = "./public_data",
    predictions_dir: str = "results/gpt/fewshot",
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
    num_examples: int = 100,
    use_mmr: bool = False,
    num_workers: int = 7,
    split: Literal["train", "validation", "dev", "test"] = "dev",
):
    llm = setup_llm(model=model)
    # Load the training data
    train_data = load_dataset(
        track="a",
        languages=[language], 
        data_root=data_root, 
        format="pandas"
    )
    if split == "dev":
        # TODO: train + validation
        train_split="train_full"
    elif split == "test":
        train_data = "train_full_with_dev"
    else:
        train_split="train"
    train_data = train_data[train_split]

    # Initialize the BGE embedding model
    retriever = get_bge_retriever(train_data, model_name, device, num_examples, use_mmr)

    chain = get_fewshot_chain(retriever, llm, language)

    os.makedirs(predictions_dir, exist_ok=True)
    predictions_file = os.path.join(predictions_dir, f"{language}_{split}_{model_name.split('/')[-1]}_{num_examples}shot_{'mmr' if use_mmr else 'cosine'}_predictions.json")

    print(f"Loading existing predictions from {predictions_file}")
    # Load existing predictions if they exist
    if os.path.exists(predictions_file):
        with open(predictions_file, "r") as f:
            predictions = json.load(f)
    else:
        predictions = {}

    data = load_dataset(
        track="a",
        languages=[language], 
        data_root=data_root, 
        format="pandas"
    )[split]
    texts_to_predict = []
    ids_to_predict = []

    # Determine which entries need predictions
    for id, text in zip(data["id"], data["text"]):
        if id not in predictions:
            texts_to_predict.append(text)
            ids_to_predict.append(id)

    # Run the chain only for entries that need predictions
    if texts_to_predict:
        chain = get_fewshot_chain(retriever, llm, language)
        results = run_chain(chain=chain, texts=texts_to_predict, ids=ids_to_predict, num_workers=num_workers)
        predictions.update(results)

        # Save updated predictions
        with open(predictions_file, "w") as f:
            json.dump(predictions, f)
    
    metrics = compute_metrics(predictions, languages=[language], split=split)
    with open(os.path.join(predictions_dir, f"{language}_{split}_{model_name.split('/')[-1]}_{num_examples}shot_{'mmr' if use_mmr else 'cosine'}_metrics.json"), "w") as f:
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

    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_id = {executor.submit(process_text, id, text): id for id, text in zip(ids, texts)}
            for future in tqdm.tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Processing texts"):
                id, result = future.result()
                if result is not None:
                    results[id] = {label: result.model_dump()[label] for label in LABELS}
    else:
        for id, text in tqdm.tqdm(zip(ids, texts), desc="Processing texts sequentially:"):
            id, result = process_text(id, text)
            if result is not None:
                results[id] = {label: result.model_dump()[label] for label in LABELS}
            else:
                results[id] = {label: 0 for label in LABELS}
    return results

def zeroshot(
    *,
    model: str,
    track: str = "a",
    language: str = "eng",
    data_root: str = "./public_data",
    predictions_dir: str = "results/gpt/zeroshot",
    num_workers: int = 7,
    split: str = "validation",
):
    llm = setup_llm(model=model)
    os.makedirs(predictions_dir, exist_ok=True)
    predictions_file = os.path.join(predictions_dir, f"{language}_{split}_predictions.json")
    
    # Load existing predictions if they exist
    if os.path.exists(predictions_file):
        with open(predictions_file, "r") as f:
            predictions = json.load(f)
    else:
        predictions = {}

    data = load_dataset(
        track="a",
        languages=[language], 
        data_root=data_root, 
        format="pandas"
    )[split]
    texts_to_predict = []
    ids_to_predict = []

    # Determine which entries need predictions
    for id, text in zip(data["id"], data["text"]):
        if id not in predictions:
            texts_to_predict.append(text)
            ids_to_predict.append(id)

    # Run the chain only for entries that need predictions
    if texts_to_predict:
        chain = get_zeroshot_chain(llm)
        results = run_chain(chain, texts_to_predict, ids_to_predict, num_workers)
        predictions.update(results)

        # Save updated predictions
        with open(predictions_file, "w") as f:
            json.dump(predictions, f)
    
    metrics = compute_metrics(predictions, languages=[language])
    with open(os.path.join(predictions_dir, f"{language}_{split}_metrics.json"), "w") as f:
        json.dump(metrics, f)


def fewshot_all_languages(
    *,
    model: str,
    data_root: str = "./public_data",
    predictions_dir: str = "results/gpt/fewshot",
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
    num_examples: int = 100,
    use_mmr: bool = False,
    num_workers: int = 7,
    split: Literal["train", "validation", "dev", "test"] = "dev",
):
    for language in tqdm.tqdm(LANGUAGES):
        print(f"Processing language {language}...")
        fewshot(
            model=model,
            data_root=data_root,
            predictions_dir=predictions_dir,
            model_name=model_name,
            device=device,
            num_examples=num_examples,
            use_mmr=use_mmr,
            num_workers=num_workers,
            split=split
        )


if __name__ == "__main__":
    fire.Fire()
