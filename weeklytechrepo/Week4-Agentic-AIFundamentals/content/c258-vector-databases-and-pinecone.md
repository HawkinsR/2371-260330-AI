# Vector Databases and Pinecone

## Learning Objectives

- Define the architectural purpose of a Vector Database in AI systems.
- Explain how Embeddings Models project semantic meaning into Dimensions.
- Calculate vector distance comparing Cosine Similarity vs Euclidean Distance.
- Execute Pinecone Setup & Configuration to host cloud vector indexes.
- Isolate distinct datasets employing Index Management & Namespaces.
- Implement standard CRUD Operations (Upsert/Query) alongside Metadata Filtering Strategies.

## Why This Matters

LLMs do not have access to your proprietary company data. Traditional relational databases (SQL) search by exact keyword matches, which performs poorly when a user asks a question using different vocabulary than the document. Vector Databases search by *meaning*, enabling Retrieval-Augmented Generation (RAG). Pinecone represents the industry standard for fast, scalable vector search in the cloud.

> **Key Term - Retrieval-Augmented Generation (RAG):** An architecture where an AI system retrieves relevant documents from a database *before* generating a response. Instead of relying solely on what the LLM learned during training, RAG grounds the response in retrieved factual context. Example: a user asks about company policy; the system retrieves the relevant policy document, then passes it to the LLM along with the question.

> **Key Term - Vector Database:** A specialized database optimized for storing and searching high-dimensional numerical vectors (embeddings). Unlike SQL which searches for exact row matches, a vector database finds the mathematically "closest" vectors using similarity calculations, enabling semantic search — finding documents by *meaning*, not exact keyword matches.

## The Concept

### Embeddings Models and Dimensions

As covered in Week 2, an embedding converts a string into an array of floating-point numbers. Modern models (like OpenAI's `text-embedding-3-small`) generate dense vectors of fixed dimensions (e.g., 1536 dimensions). This 1536-dimensional space organizes concepts mathematically so that semantically similar sentences are grouped physically close together.

### Vector Databases and Similarity

A Vector Database (like Pinecone) is highly optimized to store these massive floating-point arrays and rapidly calculate the distance between them.
When searching:

1. The user's question is embedded into a 1536-dimensional vector.
2. The Database compares this vector against all stored document vectors.
3. Using **Cosine Similarity** (which measures the angle between vectors regardless of their magnitude), it returns the mathematically "closest" documents.

> **Key Term - Cosine Similarity:** A metric measuring the angle between two vectors in a high-dimensional space. If two vectors point in nearly the same direction (angle ≈ 0°), their cosine similarity approaches 1.0 (very similar). If they point in opposite directions (angle = 180°), it approaches -1.0 (very different). Unlike Euclidean distance, cosine similarity is insensitive to the magnitude (length) of the vectors, making it ideal for comparing text embeddings.

### Pinecone Indexes, Namespaces, and Metadata

In Pinecone, an **Index** is the highest-level cluster, defined by its dimension size. To separate data within an index (e.g., "HR policies" vs "Engineering docs"), we use **Namespaces**.
Finally, vectors alone aren't enough. We attach **Metadata** (JSON objects holding things like `author`, `date`, `department`) to each vector during the "Upsert" (insert/update) phase. This allows us to perform lightning-fast hybrid searches: "Find documents semantically similar to *vacation days* BUT filter metadata strictly for `department: engineering`."

## Code Example

```python
from pinecone.grpc import PineconeGRPC as Pinecone
# Assuming PINECONE_API_KEY is in environment

# 1. Initialize Pinecone Client
pc = Pinecone()

# 2. Connect to an existing Index
# NOTE: An Index must be created first in the Pinecone console or via pc.create_index().
# The Index is defined by its dimension size (must match your embedding model's output).
# For OpenAI text-embedding-3-small, the dimension is 1536.
index = pc.Index("corporate-knowledge-base")

# Dummy Vector representing "How to file an expense report"
doc_vector = [0.1, -0.2, 0.5] + [0.0] * 1533 

# 3. Upserting Data with Metadata into a specific Namespace
index.upsert(
    vectors=[
        {"id": "doc_123", "values": doc_vector, "metadata": {"dept": "finance", "year": 2024}}
    ],
    namespace="internal-policies"
)

# Dummy Query Vector representing "I need reimbursement for travel"
query_vector = [0.15, -0.22, 0.48] + [0.0] * 1533

# 4. Querying using Cosine Similarity AND Metadata Filtering
results = index.query(
    namespace="internal-policies",
    vector=query_vector,
    top_k=1, # Return the top 1 most similar document
    include_metadata=True,
    filter={"dept": {"$eq": "finance"}} # Hard filter on metadata
)

print(f"Most relevant document ID: {results['matches'][0]['id']}")
```

## Additional Resources

- [What is a Vector Database?](https://www.pinecone.io/learn/vector-database/)
- [Pinecone Metadata Filtering Docs](https://docs.pinecone.io/guides/data/filter-with-metadata)
