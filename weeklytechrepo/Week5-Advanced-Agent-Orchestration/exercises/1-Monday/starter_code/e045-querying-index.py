import os
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# TODO: 1. Ensure you have set LANGCHAIN_TRACING_V2="true" and LANGCHAIN_API_KEY in your env

# TODO: 2. Initialize Bedrock Embeddings targeting region us-east-1
embeddings = None # Replace this

def init_vector_store():
    docs = [
        Document(page_content="LangSmith offers tracing for LLM applications.", metadata={"tool": "LangSmith"}),
        Document(page_content="Pinecone uses Namespaces to partition vector indexes.", metadata={"tool": "Pinecone"})
    ]
    # TODO: 3. Create FAISS from docs using your embeddings
    store = None
    return store

if __name__ == "__main__":
    store = init_vector_store()
    
    # TODO: 4. Perform a similarity search for "How can I partition vectors?"
    results = []
    
    for r in results:
        print(f"Content: {r.page_content}")
