# Demo: Vector Databases and Pinecone

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
| --- | --- |
| **Why SQL Fails for Semantic Search** | *"A user types 'I need some time off.' The HR database contains a document titled 'Paid Time Off Policy.' A SQL WHERE clause looks for exact text matches. Would this query find the document? What's missing?"* |
| **Embeddings / Dimensions** | *"If 'cat' becomes the vector [0.9, 0.1, 0.2] and 'kitten' becomes [0.88, 0.12, 0.22], what does the small distance between those numbers tell us about the relationship between those words?"* |
| **Cosine Similarity** | *"If two students walk in the same direction but one walks twice as fast, are they heading to different places? How does Cosine Similarity handle this difference in 'speed' (vector magnitude)?"* |
| **RAG** | *"An LLM trained in 2021 is asked: 'Who won the 2024 US election?' Why can't it answer? What architecture allows it to access information beyond its training cutoff?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/pinecone-vector-search.mermaid`.
2. Trace `Data Ingestion`. Explain that an LLM cannot natively search PDFs or Databases on a hard drive. First, we must pass the text through an Embedding Model to shatter it into an array of floats (usually 1536).
3. Trace `Retrieval`. Contrast this with traditional Search. Ask the class: "If I search SQL for 'reimbursement', will it find a document titled 'Expense Reports'?" (Answer: No, because SQL matches exact characters. Vector search matches semantic meaning, so 'reimbursement' and 'expense' end up hovering near identical X/Y/Z coordinates in the database).
4. Explain **Metadata Filters**. You cannot use Pinecone to securely isolate tenant data (Company A vs Company B) using vectors alone, because vectors hallucinate. You MUST append hard JSON attributes (metadata) to the vector so you can do hybrid searching (Semantic distance + Exact Match JSON filters).

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d056-pinecone-upsert-and-query.py`.
2. Review `get_mock_embedding()`. Show how the text is boiled down to a 3-dimensional array `[x, y, z]` based loosely on content. Financial concepts map to the X-axis (`[0.9, 0.1, 0.0]`).
3. Point out `calculate_cosine_similarity()`. You don't need to teach the math, but emphasize that Pinecone's entire multi-billion dollar valuation is based on computing this formula billions of times per second efficiently.
4. Execute the script via `run_vector_demo()`. 
5. Review Phase 1 in the terminal. Emphasize that we are saving the Mathematical Vector AND the original string `text` alongside it. Pinecone doesn't "un-embed" vectors back into text. We just give it the text back as a JSON payload attachment.
6. Review Phase 2. The user asked about a "flight". The term "flight" does not exist anywhere in the database documents! Yet, the embedding model correctly translated "flight -> reimburse -> expense array", and Pinecone found `doc1_finance` with a perfect similarity score. 

## Summary
Summarize that RAG (Retrieval-Augmented Generation) is impossible without Vector Databases. By converting messy human language into geometric arrays, we allow computers to perform algebra on human thoughts, fetching relevant context for LLMs with shocking precision.
