# Lab: Advanced Retrieval and Re-Ranking

## The Scenario
Your company's basic RAG chatbot is failing to answer complex questions that require comparing information across multiple documents. Furthermore, the vector search is returning too many irrelevant documents, clogging the LLM's context window. You need to implement an Advanced Retrieval Pipeline. First, you will build a Multi-Query router to decompose complex questions into simple sub-queries. Then, you will execute a Hybrid Search (combining Dense semantic search and Sparse keyword search). Finally, you will use a Cross-Encoder scoring function to re-rank the aggregated results, keeping only the absolute best context chunks for the final LLM payload.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e045-e045-advanced_retrieval_lab.py`.
3. Complete the `decompose_query` function:
   - Check if the `complex_query` contains BOTH `"remote"` and `"hybrid"`.
   - If it does, return a list of two distinct sub-queries (e.g., `"What is the remote work policy?"`, `"What is the hybrid work policy?"`).
   - If not, just return the original query in a list.
4. Complete the `execute_hybrid_search` function:
   - This function takes a list of `sub_queries`.
   - Iterate through each sub-query.
   - For each sub-query, call both `simulate_dense_search(sq)` and `simulate_sparse_search(sq)`.
   - Add all the returned document IDs to the `all_retrieved_docs` set to automatically deduplicate them.
   - Return the final list of deduplicated document IDs.
5. Complete the `rerank_documents` function:
   - Iterate through the provided `doc_ids`.
   - For each `doc_id`, retrieve its text content from `DUMMY_CORPUS`.
   - Simulate a cross-encoder score: Start with `score = 0`. Add `5` if `"remote"` is in the content. Add `5` if `"hybrid"` is in the content. Add `2` if `"policy"` is in the content.
   - Append a tuple of `(score, content)` to `scored_docs`.
   - Sort the list by score in descending order and return the content of the `top_k` documents.

## Definition of Done
- The script executes successfully.
- The console output proves the complex query was split into multiple sub-queries.
- The hybrid search function returns a deduplicated list of document IDs.
- The re-ranker accurately scores the documents and returns the top 2 most relevant text chunks (e.g., `doc1` and `doc2` which mention remote/hybrid policies).
