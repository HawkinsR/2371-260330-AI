# Instructor Guide: Retrieval Evaluation and RAGAS

## Overview
This demo introduces the **RAGAS (Retrieval Augmented Generation Assessment)** framework. Trainees will see how to move away from "vibes-based" evaluation and use an "LLM-as-a-Judge" to objectively score a RAG pipeline's Faithfulness and Relevancy.

## Phase 1: The Concept (Whiteboard)
**Time:** 10 mins

1.  **Open `diagrams/ragas_architecture.mermaid`**.
2.  **The Triad**: Explain that a RAG interaction consists of three parts: The **Question**, the **Context** (retrieved documents), and the **Answer**.
3.  **Faithfulness**: Explain that Faithfulness checks if the *Answer* is grounded entirely in the *Context*. If the model hallucinated, faithfulness scores drop.
4.  **Relevancy**: Explain that Relevancy checks if the *Answer* actually answers the *Question*. (You can be 100% faithful to irrelevant context, which is still a bad user experience).

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins

1.  **Open `code/d057-ragas-evaluation-pipeline.py`**.
2.  **Model Setup for RAGAS**:
    - Show how we pass our Bedrock LLM and AWS Embeddings into the RAGAS `evaluate` function. The evaluator needs an LLM to act as the "Judge".
3.  **The Dataset**:
    - Explain the `Dataset` format (from HuggingFace `datasets`). It requires columns for `question`, `answer`, and `contexts`.
4.  **Execution Display**:
    - Run the evaluation. Show how Test Case A (Accurate) receives high scores, while Test Case B (Hallucination) receives a low Faithfulness score.

## Summary Checklist for Trainees
- [ ] Do I understand the difference between Faithfulness and Answer Relevancy?
- [ ] Do I understand why RAGAS requires an LLM (LLM-as-a-Judge) to calculate these scores?
- [ ] *Note: Ragas evaluations can be slow because the judge LLM has to read and reason about every single row of the dataset.*
