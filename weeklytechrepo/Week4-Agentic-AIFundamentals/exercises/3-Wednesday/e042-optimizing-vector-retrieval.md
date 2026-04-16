# Exercise e042: Optimizing Vector Retrieval

## Overview
In this exercise, you will implement a two-stage retrieval pipeline using **Pinecone** and **Cohere Re-ranking**. You will embed documents using Amazon Bedrock's Titan model, store them in Pinecone, and then apply re-ranking to improve precision before sending context to an LLM.

## Learning Outcomes
- Embed text documents using `BedrockEmbeddings`.
- Store and query documents in a **Pinecone Serverless** index.
- Apply **Cohere Rerank** via `ContextualCompressionRetriever` to refine results.
- Understand the trade-off between recall (broad search) and precision (re-ranking).

## Prerequisites
- `PINECONE_API_KEY`, `COHERE_API_KEY`, and AWS credentials set in your environment.
- A Pinecone Serverless index created with **256 dimensions** (to match Titan v2).
- `pip install langchain-aws langchain-pinecone langchain-cohere pinecone-client`

## Instructions

Open `starter_code/e042-vector-optimization-lab.py` and complete the TODOs:

1. **Initialize Embeddings** — Use `BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")`.
2. **Create the Vector Store** — Use `PineconeVectorStore` with your index name and embedding model.
3. **Add Documents** — Create at least 5 `Document` objects with varied content and upsert them.
4. **Build the Base Retriever** — Use `.as_retriever(search_kwargs={"k": 5})`.
5. **Add the Cohere Compressor** — Use `CohereRerank(model="rerank-english-v3.0", top_n=2)` and wrap it in a `ContextualCompressionRetriever`.
6. **Run a Query** — Invoke the retriever and print the relevancy scores for the top results.

## Deliverable
Show that the re-ranked results (top 2) have higher relevancy scores than the raw top result from Pinecone alone.
