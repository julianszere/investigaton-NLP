import numpy as np
from src.datasets.LongMemEvalDataset import LongMemEvalInstance
from src.agents.RAGAgent import (
    embed_text as rag_embed_text,
    get_messages_and_embeddings as get_rag_messages_and_embeddings,
)
from src.agents.SAEAgent import (
    embed_text as sae_embed_text,
    get_messages_and_embeddings as get_sae_messages_and_embeddings,
)


def retrieve_most_relevant_messages(instance: LongMemEvalInstance, sae_embedding_model, sae_model, hook_name, rag_embedding_model_name, k: int = 10):
    rag_question_embedding = rag_embed_text(instance.question, rag_embedding_model_name)
    sae_question_embedding = sae_embed_text(sae_model, sae_embedding_model, instance.question, hook_name)
    messages, rag_embeddings, messages_time = get_rag_messages_and_embeddings(instance, rag_embedding_model_name)
    messages, sae_embeddings, messages_time = get_sae_messages_and_embeddings(instance, sae_embedding_model, sae_model, hook_name)
    rag_embeddings = np.vstack(rag_embeddings)
    sae_embeddings = np.vstack(sae_embeddings)
    rag_similarity_scores = np.dot(rag_embeddings, rag_question_embedding)
    sae_similarity_scores = np.dot(sae_embeddings, sae_question_embedding)
 
    combined_similarity_scores = np.sqrt(rag_similarity_scores**2 + sae_similarity_scores**2)
    most_relevant_messages_indices = np.argsort(combined_similarity_scores)[::-1][:k]
    most_relevant_messages = [messages[i] for i in most_relevant_messages_indices]
    time_most_relevant_messages = [messages_time[i] for i in most_relevant_messages_indices]
    return most_relevant_messages, time_most_relevant_messages


class RAGSAEAgent:
    def __init__(self, generator_model, sae_embedding_model, sae_model, hook_name, rag_embedding_model_name):
        self.generator_model = generator_model
        self.sae_embedding_model = sae_embedding_model
        self.sae_model = sae_model
        self.hook_name = hook_name
        self.rag_embedding_model_name = rag_embedding_model_name
        self.sae_embedding_model.eval()
        self.sae_model.eval()

    def answer(self, instance: LongMemEvalInstance):
        most_relevant_messages, time_most_relevant_messages = retrieve_most_relevant_messages(
            instance, self.sae_embedding_model, self.sae_model, self.hook_name, self.rag_embedding_model_name
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
