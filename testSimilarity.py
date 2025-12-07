import argparse
import json
import os
import numpy as np
from dotenv import load_dotenv
from src.models.LiteLLMModel import LiteLLMModel
from src.agents.JudgeAgent import JudgeAgent
from src.agents.RAGAgent import RAGAgent
from src.datasets.LongMemEvalDataset import LongMemEvalDataset
from config.config import Config
from transformer_lens import HookedTransformer
from sae_lens import SAE
from src.agents.SAEAgent import embed_text as sae_embed_text
from src.agents.RAGAgent import embed_text as rag_embed_text
import torch


load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Run LongMemEval evaluation pipeline")
    parser.add_argument(
        "--memory-model",
        type=str,
        default="ollama/gemma3:4b",
        help="Model name for memory/RAG agent (default: ollama/gemma3:4b)"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="ollama/gemma3:4b",
        help="Model name for judge agent (default: openai/gpt-5-mini)"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="short",
        choices=["oracle", "short"],
        help="Dataset type: oracle, short (default: short)"
    )
    parser.add_argument(
        "--dataset-set",
        type=str,
        default="longmemeval",
        choices=["longmemeval", "investigathon_evaluation", "investigathon_held_out"],
        help="Dataset set to use (default: longmemeval)"
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=10,
        help="Number of samples to process (default: 10)"
    )
    return parser.parse_args()


args = parse_args()

config = Config(
    memory_model_name=args.memory_model,
    judge_model_name=args.judge_model,
    longmemeval_dataset_type=args.dataset_type,
    longmemeval_dataset_set=args.dataset_set,
    N=args.num_samples,
)

print(f"\nInitializing models...")
print(f"  Memory Model: {config.memory_model_name}")
print(f"  Judge Model: {config.judge_model_name}")
print(f"  Embedding Model: {config.embedding_model_name}")

memory_model = LiteLLMModel(config.memory_model_name)
judge_model = LiteLLMModel(config.judge_model_name)
judge_agent = JudgeAgent(model=judge_model)
memory_agent = RAGAgent(model=memory_model, embedding_model_name=config.embedding_model_name)

longmemeval_dataset = LongMemEvalDataset(config.longmemeval_dataset_type, config.longmemeval_dataset_set)


def rag_similarity(text_1, text_2):
    z1 = memory_agent.get_embedding(message = text_1)
    z2 = memory_agent.get_embedding(message = text_2)

    z1_norm = z1 / np.linalg.norm(z1)
    z2_norm = z2 / np.linalg.norm(z2)
    return np.dot(z1_norm, z2_norm)


#################   SAE 


sae_base_model_name = "google/gemma-2b"
sae_release = "gemma-2b-res-jb"
sae_id = "blocks.17.hook_resid_post"
hook_name = "blocks.17.hook_resid_post" 

# sae_base_model_name = "EleutherAI/pythia-70m-deduped"
# sae_release = "pythia-70m-deduped-res-sm"
# sae_id = "blocks.5.hook_resid_post"
# hook_name = "blocks.5.hook_resid_post" 

device = "cuda" if torch.cuda.is_available() else "cpu"

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

def sae_similarity(text_1, text_2):
    z1 = sae_embed_text(sae_base_model, sae, text_1, hook_name)
    z2 = sae_embed_text(sae_base_model, sae, text_2, hook_name)

    z1_norm = z1 / np.linalg.norm(z1)
    z2_norm = z2 / np.linalg.norm(z2)
    return np.dot(z1_norm, z2_norm)


text_1 = 'She closed the window because the cold wind was coming in.'
text_2 = 'She closed the window because the cold wind was entering.'

text_3 = 'He rushed to board the final train that left just before twelve.'
text_4 = 'He hurried to catch the last train before midnight.'

text_5 = 'What if my child just refuses to take responsibility for their actions? How do I get them to understand the importance of being accountable?'
text_6 = 'How many days did I spend on camping trips in the United States this year?'

text_7 = 'If you pack too many tools into a small backpack, something essential will always get left behind'
text_8 = 'A neural network with fixed width can’t memorize new patterns without forgetting some of the old ones'

print("La similaridad por RAG es", rag_similarity(text_5,text_6))
print("La similaridad por SAE es", sae_similarity(text_5,text_6))