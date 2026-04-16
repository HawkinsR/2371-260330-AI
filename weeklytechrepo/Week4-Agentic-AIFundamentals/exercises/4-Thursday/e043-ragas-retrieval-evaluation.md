# Exercise 043: RAGAS Retrieval Evaluation

## Overview

In this lab, you will move from "vibe checks" to quantitative evaluation by implementing a RAGAS-style judge. You will measure the Faithfulness and Relevancy of an agent's answers based on retrieved context.

## Objectives

- Implement a mock LLM-as-a-judge for automated scoring.
- Define logic for "Faithfulness" (grounding in context).
- Automate failure alerts when hallucination is detected.

## Instructions

1. Open `starter_code/e043-ragas_lab.py`.
2. **Task 1**: Complete the `faithfulness` scoring logic to return 0.0 if the answer contains facts not found in the context.
3. **Task 2**: Implement the `answer_relevancy` logic.
4. **Task 3**: In the main pipeline, log a `[!!! CRITICAL ALERT !!!]` if the faithfulness score is below 0.5.

## Validation

- Run the script and observe "Test Case 1" (Pass).
- Observe "Test Case 2" (Hallucination) and ensure the critical alert is triggered.
