import json
import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = f"/home/ubuntu/investigaton-NLP-1/data/results/RAG/longmemeval/short/embeddings_ollama_nomic-embed-text_memory_ollama_gemma3:4b_judge_openai_gpt-5-mini"

records = []

for file in os.listdir(RESULTS_DIR):
    if file.endswith(".json"):
        with open(os.path.join(RESULTS_DIR, file), "r", encoding="utf-8") as f:
            records.append(json.load(f))
            
# --- SCORE ---
def to_bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return False   # or return None if you prefer to treat missing differently
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    # fallback: if it can't be parsed, treat as False or raise; here we choose False
    return False

scores = []
for r in records:
    if "answer_is_correct" in r:
        scores.append(to_bool(r["answer_is_correct"]))

# safety if no scores present
if len(scores) == 0:
    avg_score = float("nan")
else:
    avg_score = np.mean(scores)          # ratio True / total
    n_true = int(np.sum(scores))         # number of True
    n_total = len(scores)
    print(f"{n_true}/{n_total} = {avg_score:.3f}")
        
        
avg_score = np.mean(scores)

# --- LATENCIA ---
latencies = [r["instance_time"] for r in records]
avg_latency = np.mean(latencies)
var_latency = np.var(latencies)

# --- AVG CONTEXT LENGTH ---
def context_length(s):
    if isinstance(s, str):
        return len(s.split())   # palabras
    else:
        return 0

context_lengths = [context_length(r["relevant_context_by_RAG"]) for r in records]
avg_context_length = np.mean(context_lengths)

# --- PLOT 1: SCORE ---
plt.figure(figsize=(4,5))
plt.bar(["Score"], [avg_score])
plt.ylim(0, 1)  # score always between 0 and 1
plt.title("Average Score")
plt.ylabel("Proportion Correct")
plt.savefig("score.png")
plt.close()

# --- PLOT 2: LATENCY ---
plt.figure(figsize=(6,5))
plt.bar(["Avg Latency (s)", "Latency Var"], [avg_latency, var_latency])
plt.title("Latency Metrics")
plt.ylabel("Seconds")
plt.savefig("latency_metrics.png")
plt.close()

# --- PLOT 3: CONTEXT LENGTH ---
plt.figure(figsize=(5,5))
plt.bar(["Avg Context Length"], [avg_context_length])
plt.title("RAG Context Length")
plt.ylabel("Number of Tokens/Words")
plt.savefig("context_length.png")
plt.close()

# --- PLOT 4: HISTOGRAM ---
plt.figure(figsize=(6,4))
plt.hist(latencies, bins=20)
plt.title("Latency Distribution")
plt.xlabel("Seconds")
plt.ylabel("Count")
plt.savefig("latency_hist.png")
plt.close()
