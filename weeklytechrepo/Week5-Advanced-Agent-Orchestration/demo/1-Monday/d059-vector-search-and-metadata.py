import os
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Initialize AWS Bedrock Models
# These rely on the standard AWS environment variables or AWS profiles being configured locally
embeddings = BedrockEmbeddings(region_name="us-east-1")

def get_vector_store():
    """
    LOCAL IMPLEMENTATION (Demonstration)
    We use FAISS for local inference without needing a remote DB constraints.
    """
    docs = [
        Document(page_content="LangGraph allows cyclical agentic loops.", metadata={"topic": "langgraph"}),
        Document(page_content="Semantic Search retrieves context based on meaning.", metadata={"topic": "search"})
    ]
    # Creates an in-memory local vector store.
    store = FAISS.from_documents(docs, embeddings)
    return store

    """
    ========================================================================
    CLOUD MIGRATION PATH: MIGRATING TO PINECONE
    ========================================================================
    Once you are ready to deploy to a persistent cloud instance, uncomment 
    the following block and provide your PINECONE_API_KEY.
    
    from langchain_pinecone import PineconeVectorStore
    
    # 1. Ensure PINECONE_API_KEY is in your environment variables.
    # 2. Define the index name that you created in the Pinecone console.
    index_name = "curriculum-index"
    
    # 3. Create or connect to the Pinecone vector store using the same embeddings.
    store = PineconeVectorStore.from_documents(docs, index_name=index_name, embedding=embeddings)
    return store
    ========================================================================
    """

if __name__ == "__main__":
    print("=== Demo 059: Vector Search and Batch Metadata ===")
    
    print("Initializing local FAISS index (See code for Pinecone Migration path)...")
    store = get_vector_store()
    
    query = "How do I build cycles in my agents?"
    print(f"\nQuerying: '{query}'")
    
    results = store.similarity_search(query, k=1)
    
    print("\nResult Retrieved:")
    for res in results:
        print(f"- Content: {res.page_content} | Metadata: {res.metadata}")
