import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.datasets.LongMemEvalDataset import LongMemEvalInstance
from litellm import embedding
from sentence_transformers import SentenceTransformer

# def embed_text(message, embedding_model_name):
#     response = embedding(model=embedding_model_name, input=message)
#     return response.data[0]["embedding"]

_cached_models = {}
def embed_text(message, embedding_model_name):
    # Load the model only once
    if embedding_model_name not in _cached_models:
        _cached_models[embedding_model_name] = SentenceTransformer(
            embedding_model_name, trust_remote_code=True
        )
    model = _cached_models[embedding_model_name]
    return model.encode(message, convert_to_numpy=True)


def get_messages_and_embeddings(instance: LongMemEvalInstance, embedding_model_name):
    cache_path = f"data/rag/{instance.question_id}.parquet"
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        return df["messages"].tolist(), df["embeddings"].tolist(), df["messages_time"].tolist()

    messages = []
    embeddings = []
    messages_time = []
    for session in tqdm(instance.sessions, desc="RAG embedding sessions"):
        session_time = session.date
        for message in session.messages:
            messages.append(f"{message['role']}: {message['content']}")
            z = embed_text(message['content'], embedding_model_name)
            embeddings.append(z)
            messages_time.append(session_time)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    pd.DataFrame({"messages": messages, "embeddings": embeddings, "messages_time": messages_time}).to_parquet(cache_path)
    return messages, embeddings, messages_time


def retrieve_most_relevant_messages(instance: LongMemEvalInstance, k: int, embedding_model_name):
    question_embedding = embed_text(instance.question, embedding_model_name)
    messages, embeddings, messages_time = get_messages_and_embeddings(instance, embedding_model_name)
    similarity_scores = np.dot(embeddings, question_embedding)
    most_relevant_messages_indices = np.argsort(similarity_scores)[::-1][:k]
    most_relevant_messages = [messages[i] for i in most_relevant_messages_indices]
    time_most_relevant_messages = [messages_time[i] for i in most_relevant_messages_indices]
    return most_relevant_messages, time_most_relevant_messages


class RAGAgent:
    def __init__(self, model, embedding_model_name):
        self.model = model
        self.embedding_model_name = embedding_model_name
        
    def answer(self, instance: LongMemEvalInstance):
        most_relevant_messages, time_most_relevant_messages = retrieve_most_relevant_messages(instance, 10, self.embedding_model_name)
        evidence_blocks = []
        for msg, t in zip(most_relevant_messages, time_most_relevant_messages):
            evidence_blocks.append(f"[{t}] {msg}")
        evidence_text = "\n".join(f"{i+1}. {block}" for i, block in enumerate(evidence_blocks))
        prompt = f"""
        You are a helpful assistant that answers a question **based strictly on the provided evidence**.

        ================ Evidence ================
        {evidence_text}
        ==========================================

        The question is:
        {instance.question}

        The question was asked on:
        {instance.t_question}

        Return **only the answer**. Do not explain your reasoning.
        """
        messages = [{"role": "user", "content": prompt}]
        answer = self.model.reply(messages)
        return answer, evidence_text


