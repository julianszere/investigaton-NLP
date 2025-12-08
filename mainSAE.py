import argparse
import json
import os
from dotenv import load_dotenv
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from src.models.LiteLLMModel import LiteLLMModel
from src.agents.JudgeAgent import JudgeAgent
from src.agents.SAEAgent import SAEAgent
from src.datasets.LongMemEvalDataset import LongMemEvalDataset
from config.config import Config
import time

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
        default="openai/gpt-5-mini",
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
        default=2,
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

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nInitializing models...")
print(f"  Memory Model (generator): {config.memory_model_name}")
print(f"  Judge Model: {config.judge_model_name}")

generator_model = LiteLLMModel(config.memory_model_name)
judge_model = LiteLLMModel(config.judge_model_name)
judge_agent = JudgeAgent(model=judge_model)

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

memory_agent = SAEAgent(
    generator_model=generator_model,
    sae=sae,
    base_model=sae_base_model,
    hook_name=hook_name,
)

longmemeval_dataset = LongMemEvalDataset(config.longmemeval_dataset_type, config.longmemeval_dataset_set)

results_dir = (
    f"data/results/{config.longmemeval_dataset_set}/{config.longmemeval_dataset_type}/"
    f"SAE_{sae_release.replace('/', '_')}_{sae_id}_generator_{config.memory_model_name.replace('/', '_')}"
    f"_judge_{config.judge_model_name.replace('/', '_')}"
)

os.makedirs(results_dir, exist_ok=True)

print(f"\nResults will be saved to: {results_dir}")
print(f"Processing samples...")
print("=" * 100)

# Process samples
for instance in longmemeval_dataset[1: config.N]: #NO OLVIDARSE ESTO
    result_file = f"{results_dir}/{instance.question_id}.json"

    # if os.path.exists(result_file):
    #     print(f"Skipping {instance.question_id} because it already exists", flush=True)
    #     continue

    start_time = time.time()
    predicted_answer, predicted_relevant_messages = memory_agent.answer(instance)
    elapsed_time = time.time() - start_time

    if config.longmemeval_dataset_set != "investigathon_held_out":
        answer_is_correct = judge_agent.judge(instance, predicted_answer)

    # Save result
    with open(result_file, "w", encoding="utf-8") as f:
        result = {
            "question_id": instance.question_id,
            "question": instance.question,
            "predicted_answer": predicted_answer,
            "predicted_relevant_messages": predicted_relevant_messages,
            "elapsed_time": elapsed_time,
        }
        if config.longmemeval_dataset_set != "investigathon_held_out":
            result["answer"] = instance.answer
            result["answer_is_correct"] = answer_is_correct

        json.dump(result, f, indent=2)

        print(f"  Question: {instance.question}...")
        print(f"  Predicted: {predicted_answer}")
        if config.longmemeval_dataset_set != "investigathon_held_out":
            print(f"  Ground Truth: {instance.answer}")
            print(f"  Correct: {answer_is_correct}")
        print("-" * 100)

print("EVALUATION COMPLETE")
