import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from src.datasets.LongMemEvalDataset import LongMemEvalInstance


def sae_encode_text(model, sae, text, hook_name, prepend_bos=True):
    with torch.no_grad():
        tokens = model.to_tokens(text, prepend_bos=prepend_bos)
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        acts = cache[hook_name]
        acts_no_bos = acts[:, 1:, :]
        feature_acts = sae.encode(acts_no_bos)
        
        #Opcion 1: mean
        # z = feature_acts.mean(dim=1)
        
        #Opcioón 2: max
        z = feature_acts.max(dim=1).values
        
        return z.squeeze(0).cpu().numpy()



def load_or_compute_sae_embeddings(instance: LongMemEvalInstance, sae, model, hook_name):
    cache_path = f"data/sae_embeddings/{instance.question_id}.parquet"
    # if os.path.exists(cache_path):
    #     df = pd.read_parquet(cache_path)
    #     return df["messages"].tolist(), df["embeddings"].tolist()

    messages = []
    embeddings = []
    for session in tqdm(instance.sessions, desc="SAE embedding sessions"):
        for message in session.messages:
            messages.append(message)
            msg_text = f"{message['role']}: {message['content']}"
            z = sae_encode_text(model, sae, msg_text, hook_name)
            z = z / np.linalg.norm(z)
            embeddings.append(z)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    pd.DataFrame({"messages": messages, "embeddings": embeddings}).to_parquet(cache_path)
    return messages, embeddings


def sae_retrieve(instance: LongMemEvalInstance, sae, model, hook_name, k: int = 10):
    question_embedding = sae_encode_text(model, sae, instance.question, hook_name)
    messages, embeddings = load_or_compute_sae_embeddings(instance, sae, model, hook_name)
    embeddings = np.vstack(embeddings)
    similarity_scores = np.dot(embeddings, question_embedding)
    most_relevant_messages_indices = np.argsort(similarity_scores)[::-1][:k]
    most_relevant_messages = [messages[i] for i in most_relevant_messages_indices]
    return most_relevant_messages


class SAEAgent:
    def __init__(self, generator_model, sae, base_model, hook_name):
        self.generator_model = generator_model
        self.sae = sae
        self.base_model = base_model
        self.hook_name = hook_name
        
        self.sae.eval()
        self.base_model.eval()

    def answer(self, instance: LongMemEvalInstance):
        most_relevant_messages = sae_retrieve(
            instance,
            self.sae,
            self.base_model,
            self.hook_name,
            k=5,
        )

        prompt = f"""
        You are a helpful assistant that answers a question **based only on evidence**.
        The evidence is: {most_relevant_messages}
        The question is: {instance.question}
        Return only the answer to the question.
        """
        print(prompt)
        messages = [{"role": "user", "content": prompt}]
        answer = self.generator_model.reply(messages)
        return answer
