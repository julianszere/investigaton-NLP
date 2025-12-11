import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from src.datasets.LongMemEvalDataset import LongMemEvalInstance

def embed_text(model, sae, text, hook_name, prepend_bos=True):
    with torch.no_grad():
        tokens = model.to_tokens(text, prepend_bos=prepend_bos)
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        acts = cache[hook_name]
        acts_no_bos = acts[:, 1:, :]
        feature_acts = sae.encode(acts_no_bos)
        z = feature_acts.mean(dim=1)
        z = z.squeeze(0).cpu().numpy()
        return z / np.linalg.norm(z)


def get_messages_and_embeddings(instance: LongMemEvalInstance, sae, model, hook_name):
    cache_path = f"data/sae/{instance.question_id}.parquet"
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        return df["messages"].tolist(), df["embeddings"].tolist(), df["messages_time"].tolist()
    
    messages = []
    embeddings = []
    messages_time = []
    for session in tqdm(instance.sessions, desc="SAE embedding sessions"):
        session_time = session.date
        for message in session.messages:
            messages.append(f"{message['role']}: {message['content']}")
            z = embed_text(model, sae, message['content'], hook_name)
            embeddings.append(z)
            messages_time.append(session_time)
            
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    pd.DataFrame({"messages": messages, "embeddings": embeddings, "messages_time": messages_time}).to_parquet(cache_path)
    return messages, embeddings, messages_time


def retrieve_most_relevant_messages(instance: LongMemEvalInstance, sae, model, hook_name, k: int = 10):
    question_embedding = embed_text(model, sae, instance.question, hook_name)
    messages, embeddings, messages_time = get_messages_and_embeddings(instance, sae, model, hook_name)
    embeddings = np.vstack(embeddings)
    similarity_scores = np.dot(embeddings, question_embedding)
    most_relevant_messages_indices = np.argsort(similarity_scores)[::-1][:k]
    most_relevant_messages = [messages[i] for i in most_relevant_messages_indices]
    time_most_relevant_messages = [messages_time[i] for i in most_relevant_messages_indices]
    return most_relevant_messages, time_most_relevant_messages


class SAEAgent:
    def __init__(self, generator_model, sae, base_model, hook_name):
        self.generator_model = generator_model
        self.sae = sae
        self.base_model = base_model
        self.hook_name = hook_name
        self.sae.eval()
        self.base_model.eval()

    def answer(self, instance: LongMemEvalInstance):
        most_relevant_messages, time_most_relevant_messages = retrieve_most_relevant_messages(
            instance, self.sae, self.base_model, self.hook_name
        )
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
        answer = self.generator_model.reply(messages)
        return answer, evidence_text
