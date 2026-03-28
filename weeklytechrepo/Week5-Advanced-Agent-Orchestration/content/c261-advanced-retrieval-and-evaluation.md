# Advanced Retrieval and Evaluation

## Learning Objectives

- Architect Advanced Retrieval Patterns upgrading baseline semantic searches.
- Deconstruct complex prompts via Query Decomposition and the Multi-Query Retriever Pattern.
- Automate metadata routing through Self-Querying and Metadata Construction.
- Extract structured filters using Named Entity Recognition (NER) for Metadata Extraction.
- Configure Hybrid Search unifying Sparse (BM25) and Dense (Embedded) querying techniques.
- Improve precision by Re-ranking Retrieved Results via cross-encoder models.
- Conduct strict Retrieval Evaluation employing RAG Evaluation with RAGAS methodologies.
- Quantify vector performance defining Context Precision and Recall.

## Why This Matters

As you saw in Week 4, basic RAG is fantastic for simple queries but fails on complex ones. If a user asks, "How does the 2024 healthcare policy compare to the 2022 policy regarding vision benefits?", a single vector search will likely just return the 2024 policy and miss the comparison. We must inject Advanced Retrieval Patterns to parse, split, and evaluate the queries mathematically before they ever hit the database, ensuring we return high-precision context chunks to the LLM.

> **Key Term - Query Decomposition:** The technique of using a fast LLM to rewrite one complex user question into 2–4 simpler sub-questions, each retrieving a focused slice of context. The results are merged and deduplicated before being passed to the primary LLM. This prevents a single vector search from missing parts of a multi-part question.

## The Concept

### Multi-Query and Self-Querying

**Query Decomposition (Multi-Querying):** We use a cheap, fast LLM to intercept the user's complex question and break it into 3-4 simpler sub-queries. We embed and search *each* sub-query independently, aggregate all the unique documents returned, and pass the massive context block to the final LLM.
**Self-Querying:** Users rarely type formal metadata filters. If a user types "Show me financial reports from 2023", a Self-Querying retriever uses an LLM (and NER) to realize "2023" should not be embedded semantically, but rather converted into a strict Pinecone metadata filter: `{"year": {"$eq": 2023}}`.

> **Key Term - Self-Querying:** A retrieval pattern where an LLM automatically extracts structured metadata filters from a user's natural language query. For example, the phrase "financial reports from 2023" causes the LLM to construct a hard filter `{"year": {"$eq": 2023}}` rather than naively embedding the word "2023" as a semantic concept — preventing incorrect results.

> **Key Term - Named Entity Recognition (NER):** A Natural Language Processing technique that identifies and classifies key entities in text (such as dates, names, locations, and codes) into predefined categories. In Self-Querying retrievers, NER is used to extract structured metadata values (like years, departments, or product codes) from a user's freeform query text.

### Hybrid Search and Re-ranking

Dense embeddings (Cosine Similarity) are great for meaning, but terrible for exact keywords (like finding a specific error code "ERR-404-B"). Sparse embeddings (like BM25/TF-IDF) excel at exact keyword drops. **Hybrid Search** executes both searches simultaneously and merges the results.
Because this aggregation often returns 20+ documents, we use a **Re-ranker** (a specialized Cross-Encoder model). It individually scores how relevant each merged document is to the exact user query, and sorts them so only the absolute best 5 documents fit into the LLM's context window.

> **Key Term - Hybrid Search:** A retrieval strategy that runs a dense embedding (semantic / cosine similarity) search AND a sparse keyword search (like BM25/TF-IDF) simultaneously, merging the results. Dense search finds conceptually related documents; sparse search finds exact keyword/code matches. Hybrid search handles both scenarios better than either alone.

> **Key Term - Re-ranker (Cross-Encoder):** A specialized AI model that takes a (query, document) pair and outputs a single relevance score from 0–1. Unlike bi-encoder embeddings (which embed query and documents separately), a cross-encoder reads both together, yielding much more nuanced relevance scores. Re-rankers are applied after initial retrieval to select the top-N most relevant candidates.

### RAGAS Evaluation

How do you know if your new Re-ranker actually improved the system? You use RAGAS (Retrieval Augmented Generation Assessment). It programmatically evaluates two core retrieval metrics:

- **Context Precision:** Are the most relevant documents ranked at the very top of the retrieved list?
- **Context Recall:** Did the retriever successfully fetch *all* the information required to answer the question, or is context missing?

A practical way to remember the distinction: **Context Precision** asks "Was every chunk that was retrieved actually useful?" while **Context Recall** asks "Were all the chunks needed to answer the question actually found?" Both must be high for a production RAG system.

> **Key Term - RAGAS (Retrieval Augmented Generation Assessment):** An open-source framework for automatically evaluating RAG pipelines. RAGAS generates test questions from your documents and runs them through your pipeline, then scores the results on metrics like Context Precision (were top results truly relevant?), Context Recall (was all needed context found?), Faithfulness (did the LLM answer stick to the retrieved context rather than hallucinating?), and Answer Relevance (did the answer actually address the question?).

## Code Example

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import init_chat_model
# Assuming vectorstore is already instantiated

# 1. Initialize the cheap "Router" LLM
llm_fast = init_chat_model("gpt-3.5-turbo", model_provider="openai", temperature=0)

# 2. Implement the Multi-Query Pattern
# This automatically intercepts one complex query, asks the LLM to rewrite it
# into multiple different perspectives, searches all of them, and deduplicates the results.
advanced_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm_fast
)

# Suppose the user asks a highly complex, multi-part question:
complex_query = "What are the differences in PTO accrual rates between junior developers and senior executives according to the 2024 handbook update?"

# The `advanced_retriever` will invisibly split this into:
# 1. "PTO accrual rates junior developers 2024"
# 2. "PTO accrual rates senior executives 2024"
# 3. "2024 handbook update PTO changes"
# ... search all of them, combine the unique documents, and return them!

unique_docs = advanced_retriever.invoke(complex_query)
print(f"Aggregated {len(unique_docs)} unique documents across multiple search vectors.")
```

## Additional Resources

- [LangChain Multi-Query Retriever](https://python.langchain.com/docs/how_to/MultiQueryRetriever/)
- [RAGAS Documentation](https://docs.ragas.io/en/stable/)
