# Retrieval Evaluation and RAGAS

## Learning Objectives

- Compare Distance Metrics (Cosine, Euclidean, Dot Product) for vector search.
- Implement **Amazon Bedrock Embeddings** to generate high-quality text vectors.
- Partition and Chunk data using `RecursiveCharacterTextSplitter` and `SemanticChunking`.
- Evaluate RAG Performance with the **RAGAS** framework using key metrics.
- Apply Indexing Optimization strategies to resolve search latency and hallucination.

## Why This Matters

Building a RAG pipeline is only the first half of the battle. Without objective metrics, you cannot know if your retrieval is actually working or if the LLM is hallucinating based on irrelevant context. **RAGAS (RAG Assessment)** provides a framework for evaluating the faithfulness and relevancy of your system. Mastering retrieval evaluation ensures your AI remains grounded in truth.

## The Concept

### Distance Metrics and Similarity

When searching a vector database, we aren't looking for exact matches. We are looking for "closeness."
- **Cosine Similarity:** Measures the angle between vectors. Ideal for text where the "direction" of the meaning matters more than the "length" of the document.
- **Euclidean Distance:** Measures the straight-line distance between points.
- **Dot Product:** Measures both direction and magnitude.

### Amazon Bedrock Embeddings

To convert text into vectors, we use embedding models. **Amazon Bedrock** provides access to high-performance models like `amazon.titan-embed-text-v2:0` which supports multiple dimensions (e.g., 256, 512, 1024). Higher dimensions provide more detail but increase search latency and storage costs.

### Evaluation with RAGAS

**RAGAS (Retrieval Augmented Generation Assessment)** uses an LLM-as-a-Judge to score your system across several metrics:

1.  **Faithfulness:** Measures if the answer is derived *only* from the retrieved context. This is the primary defense against hallucinations.
2.  **Answer Relevancy:** Measures how well the answer addresses the user's question, regardless of whether it's grounded in context.
3.  **Context Precision:** Measures the signal-to-noise ratio in the retrieved documents.
4.  **Context Recall:** Measures if all necessary information to answer the question was actually retrieved.

> **Key Term - RAGAS Framework:** An evaluation library designed specifically for RAG pipelines. It provides a more granular look at where the pipeline is failing (e.g., "Good retrieval, but bad generation" vs. "Bad retrieval leading to hallucinations").

## Code Example

```python
from langchain_aws import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 1. Initialize Bedrock Embeddings
# We use Titan V2 for efficient, high-quality vector generation
embeddings_model = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1"
)

# 2. Document Loading & Chunking
loader = PyPDFLoader("employee_handbook.pdf")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = loader.load_and_split(text_splitter)

# 3. Create Local Vector Store (FAISS)
vectorstore = FAISS.from_documents(docs, embeddings_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Usage in a RAG Logic (Simplified RAGAS check)
query = "What is the policy on corporate travel?"
retrieved_docs = retriever.invoke(query)

# In a real RAGAS pipeline, you would pass (query, context, answer) 
# to the Ragas evaluator to receive scores from 0 to 1.
for i, d in enumerate(retrieved_docs):
    print(f"Chunk {i+1} Relevancy Check: {d.page_content[:150]}...")
```

## Additional Resources

- [Amazon Bedrock Titan Embeddings](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html)
- [RAGAS Documentation](https://docs.ragas.io/en/stable/)
- [FAISS Vector Store in LangChain](https://python.langchain.com/docs/integrations/vectorstores/faiss/)
