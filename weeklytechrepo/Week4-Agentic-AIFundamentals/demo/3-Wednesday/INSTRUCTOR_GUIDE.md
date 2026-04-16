# Instructor Guide: Vector DBs and Context Optimization

## Overview
This demo illustrates the leap from basic semantic search to production-grade **Context Optimization** using Pinecone and Cohere. Trainees will see why vector similarity alone is insufficient for high accuracy.

## Phase 1: The Concept (Whiteboard)
**Time:** 10 mins

1.  **Open `diagrams/retrieval_optimization.mermaid`**.
2.  **Recall (Pinecone)**: Explain that the first step retrieves a broad set of candidates (e.g., top 10). It's fast but can be noisy.
3.  **Rerank (Cohere)**: Explain that the second step takes those top 10 and passes them to a specialized cross-encoder model to score them precisely, returning the true top 3.
4.  **Discussion**: Ask: "Why not just re-rank the entire database?" (Answer: Re-ranking is slow and expensive. We use fast vector search to narrow it down, then re-rank the shortlist).

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins

1.  **Open `code/d056-pinecone-vector-management.py`**.
2.  **Vector Store Setup**:
    - Highlight the `PineconeVectorStore` initialization.
    - Mention that namespaces isolate data natively.
3.  **The Base Retriever**:
    - Show `search_kwargs={"k": 10}`. Emphasize we grab more than we need.
4.  **The Compressor (Re-ranker)**:
    - Introduce `CohereRerank`. Show how it acts as a filter over the base retriever using `ContextualCompressionRetriever`.
5.  **Execution Display**:
    - Run the script (requires `PINECONE_API_KEY` and `COHERE_API_KEY`).
    - Point out how the final output only shows the top 3 results, with exact relevancy scores attached.

## Summary Checklist for Trainees
- [ ] Do I understand the difference between Recall (fetching) and Precision (re-ranking)?
- [ ] Are my Pinecone and Cohere API keys set?
- [ ] Does my Pinecone index dimension match my embedding model (e.g., 256 for Titan v2)?
