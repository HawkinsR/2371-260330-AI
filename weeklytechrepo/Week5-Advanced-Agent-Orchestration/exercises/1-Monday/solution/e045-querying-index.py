import os
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# SOLUTION 1: Enable LangSmith Tracing via environment
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "e045-vector-search-lab"

# SOLUTION 2: Initialize Bedrock Embeddings
embeddings = BedrockEmbeddings(region_name="us-east-1")

def init_vector_store():
    docs = [
        Document(page_content="LangSmith offers tracing for LLM applications.", metadata={"tool": "LangSmith"}),
        Document(page_content="Pinecone uses Namespaces to partition vector indexes.", metadata={"tool": "Pinecone"})
    ]
    # SOLUTION 3: Create FAISS from docs
    store = FAISS.from_documents(docs, embeddings)
    return store

if __name__ == "__main__":
    store = init_vector_store()
    
    # SOLUTION 4: Perform search
    query = "How can I partition vector data logically?"
    print(f"Query: {query}")
    results = store.similarity_search(query, k=1)
    
    print("\nResult Retrieved:")
    for r in results:
        print(f"Content: {r.page_content}")
    
    print("\n[Check your LangSmith console to view the Execution Trace!]")
