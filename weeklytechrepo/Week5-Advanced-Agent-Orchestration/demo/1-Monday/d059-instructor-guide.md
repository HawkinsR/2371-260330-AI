# Demo: Advanced Retrieval and Evaluation

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
| --- | --- |
| **Query Decomposition** | *"A user asks: 'Compare the PTO policy for junior vs. senior employees AND tell me the 2023 insurance deductible.' How many separate retrieval searches should you execute? Why does wrapping this in one single vector query fail?"* |
| **Hybrid Search** | *"When would dense (semantic) search fail, but sparse (keyword) search succeed? Give a concrete example with something a developer would search for."* |
| **Re-Ranker** | *"After Multi-Query + Hybrid Search, you have 30 candidate documents but can only pass 5 to the LLM. The Re-Ranker job-ranks them. What's a faster but less accurate alternative to a cross-encoder for this re-ranking step?"* |
| **RAGAS Evaluation** | *"How would you know if your new Re-Ranker actually improved retrieval quality without RAGAS? What are the downsides of manual spot-checking at scale?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/multi-query-hybrid-search.mermaid`.
2. Trace the path from the Complex User Query down. Ask the class: "Why does the Multi-Query LLM break one query into three?" (Answer: Vector spaces are mathematical. A prompt comparing 2022 and 2024 exists in a very specific mathematical space. By breaking it down, we ensure we calculate similarities for the 2022 portion individually from the 2024 portion).
3. Walk through the **Hybrid Search Engine**. Emphasize that Dense = Concepts (e.g., "vacation" matches "PTO") and Sparse = Exact Strings (e.g., matching the exact error code ERR-404-B).
4. Explain the Re-ranker step. If 3 sub-queries each pull 5 documents from Dense and 5 from Sparse, you now have 30 documents. You cannot stuff 30 documents into an LLM context efficiently. The Cross-Encoder acts as a gatekeeper to rank the top 5.

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d059-multi-query-and-hybrid-search.py`.
2. Review the `simulate_llm_query_decomposition()` function. Show how it takes a complex string and creates an array of smaller string targets.
3. Review the Pipeline. Point out `all_retrieved_docs = set()`. This is a classic Python trick for automated deduplication. If Sparse search and Dense search find the identical `doc4`, the set ensures it is only passed to the Re-ranker once.
4. Execute the script via `demonstrate_advanced_retrieval()`.
5. Point out the [Re-ranker] logs. Show how it scores 'doc1' (healthcare) lower than the PTO documents because it realizes the original complex query was explicitly asking about PTO. 

## Summary
Reiterate that production RAG is not simply calling `similarity_search()`. It is a multi-step ETL pipeline optimizing the exact strings fed into the final language model. Evaluated against RAGAS metrics, multi-query hybrid pipelines drastically outperform zero-shot retrieval.
