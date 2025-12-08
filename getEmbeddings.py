import argparse
import json
import os
from dotenv import load_dotenv
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from src.models.LiteLLMModel import LiteLLMModel
from src.agents.JudgeAgent import JudgeAgent
from src.agents.SAEAgent import get_messages_and_embeddings
from src.datasets.LongMemEvalDataset import LongMemEvalDataset
import time


load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

sae_base_model_name = "gemma-2b-it"
sae_release = "gemma-2b-it-res-jb"
sae_id = "blocks.12.hook_resid_post"
hook_name = "blocks.12.hook_resid_post" 

print(f"  SAE Base Model: {sae_base_model_name}")
print(f"  SAE Release: {sae_release}")
print(f"  SAE ID: {sae_id}")

sae_base_model = HookedTransformer.from_pretrained_no_processing(
    sae_base_model_name,
    device=device,
    dtype=torch.bfloat16,
)

sae, sae_cfg, sparsity = SAE.from_pretrained(
    release=sae_release,
    sae_id=sae_id,
    device=device,
)
sae.eval()


longmemeval_dataset = LongMemEvalDataset("short", "longmemeval")


print(f"Processing samples...")
print("=" * 100)

# Process samples
for instance in longmemeval_dataset[0:1]:
    
    start_time = time.time()
    predicted_answer, predicted_relevant_messages = memory_agent.answer(instance)
    elapsed_time = time.time() - start_time


    # RESPUESTAS 
    oracle_answers = {
        "oracle_SAE_similarity": [],
        "oracle_RAG_similarity": []
    }
    
    # RESPUESTAS 
    no_oracle_answers = {
        "dates": [],
        "contents": [],
        "no_oracle_SAE_similarity": [],
        "no_oracle_RAG_similarity": []
    }

    for sess in tqdm(instance.sessions, desc="Sessions"):

        # CASE 1: This session DOES have at least one answered message
        if any(msg.get("has_answer", False) for msg in sess.messages):

            for m in sess.messages:
                if m.get("has_answer", False):
                    
                    oracle_answers["dates"].append(sess.date)
                    oracle_answers["contents"].append(m["content"])
                    
                    oracle_answers["oracle_RAG_similarity"].append(rag_similarity(m["content"], instance.question))
                    
                    oracle_answers["oracle_SAE_similarity"].append(sae_similarity(m["content"], instance.question))

        # CASE 2: No answered messages at all
        else:
            for m in sess.messages:
                
                no_oracle_answers["dates"].append(sess.date)
                no_oracle_answers["contents"].append(m["content"])
                
                no_oracle_answers["no_oracle_RAG_similarity"].append(rag_similarity(m["content"], instance.question))
                    
                no_oracle_answers["no_oracle_SAE_similarity"].append(sae_similarity(m["content"], instance.question))

    # SAVE THIS INSTANCE’S RESULTS
    results.append({
        "question": question,
        "t_question": t_question,
        "answer": answer,
        "oracle_answers": oracle_answers,
        "no_oracle_answers": no_oracle_answers
    })

print("EVALUATION COMPLETE")