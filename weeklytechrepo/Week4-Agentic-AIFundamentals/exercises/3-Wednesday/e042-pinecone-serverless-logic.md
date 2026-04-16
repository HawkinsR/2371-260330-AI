# Exercise 042: Pinecone Serverless Logic

## Overview

In this lab, you will implement the storage and retrieval side of an agentic workflow using the Pinecone Serverless architecture. You will also use LangSmith-style logging to trace the latency of your cloud operations.

## Objectives

- Implement an upsert function for Pinecone Serverless.
- Use metadata filters to prune search results by department.
- Implement a mock latency logger (LangSmith simulation).

## Instructions

1. Open `starter_code/e042-pinecone_lab.py`.
2. **Task 1**: Complete the `upsert` method in `MockPineconeServerless`.
3. **Task 2**: Implement the `log_to_langsmith` function to calculate and print operation latency.
4. **Task 3**: Use the `filters` argument in the `query` method to restrict results to the "HR" department.

## Validation

- Run the script.
- Verify that the upsert and query actions are followed by a `[LANGSMITH TRACE]`.
- Check that the final retrieved text correctly identifies the "Vacation policy".
