# SAE-Based RAG for LongMemEval / Investigathon

This repository implements a memory retrieval pipeline for long-horizon conversational QA, with a focus on a **Sparse Autoencoder (SAE) based Retrieval-Augmented Generation (RAG)** setup.

The project works on LongMemEval-style instances: each example contains many past chat sessions, a final question, and optionally a ground-truth answer. The goal is to recover the most relevant past messages and answer the question using only that retrieved evidence.

The main idea in this repo is a **hybrid retriever**:

- A standard semantic retriever embeds each message with a sentence embedding model.
- An SAE retriever embeds each message using activations from a transformer layer passed through a sparse autoencoder.
- The final `RAGSAE` pipeline combines both similarity scores and retrieves the top messages for generation.

This makes the repository useful both as:

- a baseline semantic RAG system
- a pure SAE retrieval system
- a hybrid SAE + semantic RAG system

For the benchmark/task definition, evaluation protocol, and dataset details, see `benchmark_explanation.md`.

## Repository Overview

The most important entrypoints are:

- `scripts/RAG_retrieval.py`: semantic RAG baseline
- `scripts/SAE_retrieval.py`: retrieval using only SAE-based representations
- `scripts/RAG_SAE_retrieval.py`: hybrid retriever that combines semantic and SAE similarity

Core modules:

- `src/agents/RAG.py`: semantic retrieval over conversation messages
- `src/agents/SAE.py`: SAE-based retrieval over conversation messages
- `src/agents/RAG_SAE.py`: combined retrieval score from both systems
- `src/agents/Judge.py`: LLM-as-a-judge evaluation for non-held-out sets
- `src/datasets/LongMemEvalDataset.py`: dataset loader for LongMemEval and Investigathon splits
- `src/models/LiteLLMModel.py`: unified generation/judge model wrapper via LiteLLM
- `config/config.py`: experiment configuration

## How the SAE-Based RAG Pipeline Works

`scripts/RAG_SAE_retrieval.py` is the main pipeline if you want the SAE-based RAG version.

For each benchmark instance, it does the following:

1. Loads all conversation sessions associated with the question.
2. Builds one embedding per message using a standard text embedding model.
3. Builds another embedding per message using:
   - `gemma-2b-it` as the base transformer
   - the SAE release `gemma-2b-it-res-jb`
   - activations from `blocks.12.hook_resid_post`
4. Embeds the question with both methods.
5. Computes message relevance with both similarity scores.
6. Combines the two scores with:

```text
combined_score = sqrt(rag_score^2 + sae_score^2)
```

7. Retrieves the top messages.
8. Sends only those retrieved messages as evidence to the generator model.
9. Optionally evaluates the answer with a judge model when ground truth is available.

The generator is therefore not reading the full conversation history. It answers from a compressed set of retrieved evidence.

## Retrieval Variants in This Repo

### 1. Semantic RAG

`scripts/RAG_retrieval.py` uses a sentence embedding model to index messages and rank them by similarity to the question.

Default embedding model:

- `nomic-ai/nomic-embed-text-v1`

This is the simplest baseline in the repository.

### 2. SAE Retrieval

`scripts/SAE_retrieval.py` replaces semantic embeddings with SAE-based representations extracted from transformer activations.

This lets you test whether sparse internal features are better memory keys than standard embedding vectors.

### 3. Hybrid SAE + RAG

`scripts/RAG_SAE_retrieval.py` combines both retrieval signals and is the closest thing to the project's main idea.

If you want the README answer to "what is this project doing?", this is it:

**it is a long-memory QA system that retrieves evidence from past conversations using both semantic embeddings and SAE feature embeddings, then generates an answer from the retrieved evidence.**

## Datasets

The dataset loader supports three dataset sets:

- `longmemeval`
- `investigathon_evaluation`
- `investigathon_held_out`

And these dataset types:

- `oracle`
- `short`

Notes:

- `oracle` contains only the relevant sessions for a question.
- `short` contains the full conversational haystack.
- The held-out split only supports `short`.

The loader reads:

- `data/longmemeval/longmemeval_oracle.json`
- `data/longmemeval/longmemeval_s_cleaned.json`
- `data/investigathon/Investigathon_LLMTrack_Evaluation_oracle.json`
- `data/investigathon/Investigathon_LLMTrack_Evaluation_s_cleaned.json`
- `data/investigathon/Investigathon_LLMTrack_HeldOut_s_cleaned.json`

## Setup

This project is configured with `uv`.

### 1. Install dependencies

```sh
uv sync
```

### 2. Download datasets

```sh
uv run scripts/download_dataset.py
```

## Model Requirements

The code uses three kinds of models:

- a generator model for answering questions
- a judge model for evaluation
- retrieval models for embeddings / SAE activations

Defaults in the scripts:

- Generator: `ollama/gemma3:4b`
- Judge: `openai/gpt-5-mini`
- Semantic embedding model: `nomic-ai/nomic-embed-text-v1`
- SAE base model: `gemma-2b-it`
- SAE release: `gemma-2b-it-res-jb`
- SAE hook: `blocks.12.hook_resid_post`

## Running the Pipelines

### Semantic RAG baseline

```sh
uv run python scripts/RAG_retrieval.py --dataset-set investigathon_evaluation --dataset-type short --num-samples 250
```

### Pure SAE retrieval

```sh
uv run python scripts/SAE_retrieval.py --dataset-set investigathon_evaluation --dataset-type short --num-samples 250
```

### Hybrid SAE-based RAG

```sh
uv run python scripts/RAG_SAE_retrieval.py --dataset-set investigathon_evaluation --dataset-type short --num-samples 250
```

### Held-out predictions

```sh
uv run python scripts/RAG_SAE_retrieval.py --dataset-set investigathon_held_out --dataset-type short --num-samples 250
```

## Outputs

Results are written under `data/results/` with separate folders for:

- `RAG/`
- `SAE/`
- `RAGSAE/`

Depending on the script, each result file can include:

- `question_id`
- `question`
- `predicted_answer`
- `predicted_relevant_messages`
- `elapsed_time`
- `answer`
- `answer_is_correct`

The code also caches message embeddings to avoid recomputing them:

- semantic cache: `data/rag/<question_id>.parquet`
- SAE cache: `data/sae/<question_id>.parquet`

## Evaluation

For non-held-out datasets, answers are evaluated with `Judge`, which uses an LLM as a binary correctness judge against the reference answer.

For the held-out dataset, no gold answer is available, so the scripts only save predictions.

## What This Project Is Really About

At a high level, this repository is not just a generic benchmark starter. It is an experiment in **memory retrieval for long conversations**, where the key hypothesis is:

**SAE-derived features may retrieve useful evidence differently from standard semantic embeddings, and combining both may improve long-context memory retrieval.**

If you are presenting this project, a concise description would be:

> A hybrid SAE-based RAG system for long-memory conversational question answering on LongMemEval/Investigathon data.

## Known Caveats

- Some helper files in the repo look exploratory rather than production-ready.
- The judge uses `eval()` on the model output, so malformed judge responses can break evaluation.
- `scripts/RAG_SAE_retrieval.py` currently saves a reduced output schema compared with `scripts/RAG_retrieval.py` and `scripts/SAE_retrieval.py`.
