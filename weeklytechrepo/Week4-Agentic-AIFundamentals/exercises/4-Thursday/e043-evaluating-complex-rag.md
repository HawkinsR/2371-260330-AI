# Exercise e043: Evaluating Complex RAG with RAGAS

## Overview
In this exercise, you will build a complete RAG evaluation pipeline using the **RAGAS** framework with **Amazon Bedrock** as the LLM judge. You will evaluate two contrasting RAG outputs — a grounded answer and a hallucinated answer — and interpret the resulting metrics.

## Learning Outcomes
- Structure a RAGAS evaluation dataset with `question`, `answer`, and `contexts` columns.
- Use `BedrockEmbeddings` and `ChatBedrock` as the LLM-as-a-Judge models.
- Call `ragas.evaluate()` with `faithfulness` and `answer_relevancy` metrics.
- Interpret **Faithfulness** vs. **Answer Relevancy** scores to diagnose RAG failures.

## Prerequisites
- AWS credentials set in environment with Bedrock access.
- `pip install ragas datasets langchain-aws`

## Instructions

Open `starter_code/e043-ragas-evaluation-lab.py` and complete the TODOs:

1. **Define the Judge Models** — Initialize `ChatBedrock` (Claude 3.5 Sonnet) and `BedrockEmbeddings` (Titan v2).
2. **Build the Dataset** — Create a `datasets.Dataset` with at least 2 rows:
   - Row A: A faithful answer (derived from the context).
   - Row B: A hallucinated answer (claims something not in the context).
3. **Run the Evaluation** — Call `ragas.evaluate()` passing your dataset, metrics, `llm`, and `embeddings`.
4. **Analyze Results** — Convert to a Pandas DataFrame and print the scores for each row.

## Deliverable
Demonstrate that Row A (faithful) achieves a **Faithfulness score ≥ 0.9**, while Row B (hallucinated) achieves a **Faithfulness score ≤ 0.2**.
