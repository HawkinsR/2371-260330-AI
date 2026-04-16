# Exercise 044: Re-ranking and Context Optimization

## Overview

In this final lab, you will optimize the context fed to an LLM by implementing a Cohere-style re-ranker and adding security guardrails to protect against "context stuffing" or malicious prompt injections hidden in retrieved documents.

## Objectives

- Implement a cross-encoder re-ranking step.
- Prune low-relevance context chunks to save tokens and improve accuracy.
- Use security guardrails to scan for injection attempts.

## Instructions

1. Open `starter_code/e044-reranking_security_lab.py`.
2. **Task 1**: Implement the `rerank` logic in `MockCohereReranker` using a query-keyword intersection score.
3. **Task 2**: Prune documentation chunks that fall below a 0.5 relevance threshold.
4. **Task 3**: Complete the `validate_context` guardrail to detect forbidden phrases like "ignore previous instructions".

## Validation

- Run the script.
- Review "Scenario A" and confirm that the malicious injection is caught by the guardrail.
- Review "Scenario B" and verify that only relevant chunks remain in the final context.
