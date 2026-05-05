# Explanation of the Benchmark and Evaluation Criteria for the YHat Investigathon

## Introduction

LongMemEval is a benchmark designed to evaluate long-term memory systems in conversational assistants. Unlike classic QA tasks, the focus here is on measuring **whether a system can remember, update, synthesize, and retrieve information dispersed across long histories**.

In this document we explain:

- How the original LongMemEval benchmark is formulated.  
- What skills it measures and how instances are constructed.  
- How we evaluate in this Investigathon track, including our *own benchmark extension* with new questions.  
- What teams must submit and how they will be evaluated.

---

## Version of the Benchmark Used

In this competition we will use the **S version** of LongMemEval, which contains a sequence of sessions totaling ~115k tokens.

### Formulation

Each benchmark instance is a **4-tuple**:

\[
(S, q, a)
\]

where:

- **S** is a sequence of sessions ordered chronologically:  

  \[
  S \equiv [(t_1, S_1), (t_2, S_2), ..., (t_N, S_N)]
  \]

- Each **Sᵢ** is a multi-turn interaction between the user and the assistant. Each message has a timestamp.  
- Each session can be decomposed into *rounds*: one user message followed by one assistant message.  
- **q** is the final question.  
- **a** is the correct answer (short and concise).

### How is it evaluated?

- The system receives the full history `S`, which it must process somehow (e.g., using RAG or any approach you design).  
- Then it is given the question `q`.  
- It must generate an answer that will be evaluated by an LLM (see Metrics section).

---

## What LongMemEval Measures

The benchmark evaluates five fundamental abilities:

### **1. Information Extraction (IE)**  
Recall specific details from the history, whether stated by the user or the assistant.

### **2. Multi-Session Reasoning (MR)**  
Integrate information across different sessions to answer questions requiring synthesis.

### **3. Knowledge Updates (KU)**  
Detect and update user information as it changes over time.

### **4. Temporal Reasoning (TR)**  
Reason about dates, sequences, and temporally ordered events.

### **5. Abstention (ABS)**  
Recognize when a question cannot be answered with the available information and return "I don’t know".

---

## Question Types

LongMemEval generates seven main categories:

- **Single-session-user**  
- **Single-session-assistant**  
- **Single-session-preference**  
- **Multi-session** (MR)  
- **Knowledge-update** (KU)  
- **Temporal-reasoning** (TR)  
- **Abstention** (30 questions designed to measure non-hallucination)

Each category captures a different expected behavior of a memory-capable assistant.

---

## How the Original Benchmark is Built

The benchmark defines 164 attributes organized into:

- lifestyle  
- belongings  
- life events  
- situation context  
- demographic information  

### Background sampling  
For each attribute, an LLM generates a paragraph written from the user’s perspective.

### QA generation  
From each paragraph, another model generates (question, answer) pairs.  
These are then reviewed by humans for quality and diversity.

### Evidence Session Construction  
The authors generate additional sessions containing the evidence needed to answer the questions, but distributed and mixed with realistic conversational noise.

### History Compilation  
All sessions are assembled in temporal order, forming long and complex histories.

---

## Benchmark Metrics

Since answers are open-ended, exact match is not used.  
The benchmark uses **LLM-as-a-judge**. You must use the same prompt provided in the repository.

---

# Model Restrictions

Each team may use **any model up to 4B parameters** for any part of the system.

This includes:

- Local models (Qwen3-4B, Gemma-3-4B, etc.)

The goal is to evaluate **memory and efficiency**, not brute force or large models.

---

# Special Investigathon Benchmark (Very Important)

For this track, in addition to the official benchmark, **we generated an additional set** with 500 new questions using the original histories:

### ✔ 250 new questions with answers  
You can use these as an evaluation set for your system.

### ✔ 250 additional questions without answers  
This is the held-out set that we will use to evaluate your system.

---

### Mandatory Submission

You must submit a file with answers for these 250 questions:

**📅 Deadline:**  
**December 11 at 16:00 (24 hours before the final on December 12)**  

Details on how to submit will be sent by email next week.

---

### Evaluation

We will evaluate automatically using **GPT-5-mini** with the same `Judge` prompt included in the repository.

We recommend using the same model for your own evaluation.

---

# What Teams Must Report

Your results must include at least the following metrics:

### 1. Score  
Average accuracy according to the LLM judge.

### 2. Latency  
Average time per question.

### 3. Latency Variance  
Variance of latency across experiments.

### 4. Average Context Length  
Average length of the context sent to the model per question.

This allows comparison between:
- Retrieval-heavy methods (RAG)  
- Compression or dynamic summarization approaches  

Include these metrics in your tables and plots.

---

# Baseline

In addition to reporting main metrics, each team must include a comparison against a standard **Retrieval-Augmented Generation (RAG)** baseline under the same constraints (≤ 4B model).

This repository provides an implementation of this RAG baseline. Instructions are available in the README.

The organizers are not responsible for errors in the provided RAG implementation. Teams are expected to understand and verify the code they use.

---

# Evaluation Criteria

Beyond final performance on the held-out set, teams will also be evaluated on the **entire research process**, including clarity, rigor, and creativity of ideas.
