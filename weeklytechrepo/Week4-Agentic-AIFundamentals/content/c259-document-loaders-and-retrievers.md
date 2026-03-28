# Document Loaders and Retrievers

## Learning Objectives

- Extract unformatted data streams utilizing LangChain Document Loaders (PDF, Web, Text).
- Prevent context window overflow employing recursive Text Splitting Techniques.
- Institute the `create_retriever` standard implementation pipeline.
- Differentiate semantic Vector Search limits versus exact Keyword Search.
- Configure Custom Retriever Logic to inject multi-modal data safely.
- Appraise strategies for thorough Indexing Optimization resolving search latency.

## Why This Matters

Vector databases are useless if they are empty. The process of ingesting raw unstructured data (PDFs, Web pages, Word Docs) and meticulously processing it so the LLM can properly digest it is often the hardest part of building AI systems. A poorly parsed PDF will result in misaligned vectors, degrading search retrieval quality significantly. Clean ingestion and optimized chunking are the backbone of a high-functioning AI retrieval pipeline.

## The Concept

### Document Loaders

LangChain provides hundreds of Document Loaders. Whether the source is a local `.txt` file, a remote S3 bucket, a raw HTML web page, or a massive PDF, a loader handles the complex parsing natively, returning standard LangChain `Document` objects containing the raw `page_content` string and associated `metadata` (like the source hidden URL).

> **Key Term - Document Loader:** A software component that reads a raw data source (PDF, website, database, S3) and converts it into a standardized `Document` object containing the text (`page_content`) and origin metadata. Document loaders abstract the messy, format-specific parsing logic so the rest of the pipeline can work with uniform objects regardless of source type.

### Text Splitting (Chunking)

LLMs have strict limits on how much text they can read at once (the Context Window). We cannot embed an entire 500-page PDF into a single vector. We must split the document into smaller "chunks" (e.g., 1000 characters per chunk).
The `RecursiveCharacterTextSplitter` is the industry standard. It attempts to split by paragraphs; if the paragraph is still too big, it splits by sentences, ensuring semantic ideas are rarely cut in half. We also employ "Chunk Overlap" (e.g., chunk 2 starts 100 characters before chunk 1 ends) to ensure context isn't lost at the rigid boundaries.

> **Key Term - Chunking (Text Splitting):** The process of dividing a large document into smaller, fixed-size pieces ("chunks") before embedding and indexing. Each chunk is embedded as its own vector, ensuring granular retrieval. Too-large chunks waste context window space; too-small chunks lose semantic meaning. Typical chunk sizes are 200–1000 characters.

> **Key Term - Chunk Overlap:** A technique where consecutive chunks share a portion of their text (e.g., chunk 2 starts 100 characters before chunk 1 ends). This prevents important context from being lost when a sentence or idea crosses a chunk boundary, at the cost of slightly higher storage.

### Retrievers

A Retriever is a high-level LangChain interface that wraps our Vector Database. While you could manually write the embedding and Pinecone query logic (as seen previously), calling `vectorstore.as_retriever()` simplifies the process immediately. It exposes clean methods (like `invoke("search query")`) that automatically handle the query embedding, execute the similarity search, and return the top formatted `Document` objects.

> **Key Term - Retriever:** A LangChain abstraction that accepts a text query string and returns the most relevant `Document` objects from a backing store (vector database, keyword search engine, etc.). Retrievers hide the complexity of embedding the query and searching the database, providing a single `invoke(query)` interface regardless of the underlying search strategy.

## Code Example

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS # Local memory vectorstore example

# 1. Document Loading
loader = PyPDFLoader("sample_handbook.pdf")
raw_docs = loader.load()
print(f"Loaded {len(raw_docs)} raw pages.")

# 2. Text Splitting (Chunking)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Max characters per chunk
    chunk_overlap=50 # Number of characters to overlap between chunks
)
split_docs = text_splitter.split_documents(raw_docs)
print(f"Split into {len(split_docs)} semantic chunks.")

# 3. Embedding and Indexing (Creating the Vector Database)
embeddings_model = OpenAIEmbeddings()
# FAISS acts exactly like Pinecone, but totally local in RAM for prototyping
vectorstore = FAISS.from_documents(split_docs, embeddings_model)

# 4. Creating the Retriever Interface
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks

# 5. Usage
relevant_docs = retriever.invoke("What is the company vacation policy?")
for i, doc in enumerate(relevant_docs):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
```

## Additional Resources

- [LangChain Text Splitters Tutorial](https://python.langchain.com/docs/concepts/text_splitters/)
- [LangChain Retrievers Documentation](https://python.langchain.com/docs/concepts/retrievers/)
