import os
from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from pinecone import Pinecone, ServerlessSpec

# =====================================================================
# 1. Environment and Constants Setup
# =====================================================================
# Ensure these are set in your environment:
# PINECONE_API_KEY, COHERE_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
INDEX_NAME = "corp-knowledge-v4"
NAMESPACE = "internal-policies"

# =====================================================================
# 2. Pipeline Initialization
# =====================================================================
def get_optimized_retriever():
    print("   [SETUP]: Initializing Amazon Titan Embeddings...")
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")

    print(f"   [SETUP]: Connecting to Pinecone Index '{INDEX_NAME}'...")
    pc = Pinecone() # Uses PINECONE_API_KEY from env
    
    # 1. Base Retriever: Fast Vector Search (Recall)
    # We use Pinecone to quickly find the top 10 candidates based on cosine similarity
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embeddings, 
        namespace=NAMESPACE
    )
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 2. Compressor: Precision Re-ranking
    # We use Cohere to re-score the top 10 and return only the highest quality top 3
    print("   [SETUP]: Initializing Cohere Re-ranker compressor...")
    compressor = CohereRerank(model="rerank-english-v3.0", top_n=3)

    # 3. Combine into an Optimized Pipeline
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    return compression_retriever, vectorstore

# =====================================================================
# 3. Execution (Data Loading and Optimization)
# =====================================================================
def run_demo():
    print("=== Week 4: Context Optimization (Pinecone + Cohere) ===")
    
    try:
        retriever, vectorstore = get_optimized_retriever()
    except Exception as e:
        print(f"\n[ERROR]: Failed to connect to services. Ensure AWS, Pinecone, and Cohere keys are set. Details: {e}")
        return

    # --- Step 1: Simulate Data Ingestion ---
    print("\n--- PHASE 1: Data Indexing (Upsert) ---")
    docs = [
        Document(page_content="Traveling securely entails avoiding public Wi-Fi.", metadata={"topic": "security"}),
        Document(page_content="Employees get 20 days of Paid Time Off (PTO) per year.", metadata={"topic": "hr"}),
        Document(page_content="The VPN must be active to access the HR portal.", metadata={"topic": "it"})
    ]
    print(f"[PINECONE]: Upserting {len(docs)} documents into '{NAMESPACE}'...")
    # NOTE: In reality, you'd only run this once.
    # vectorstore.add_documents(docs)

    # --- Step 2: Optimized Retrieval ---
    print("\n--- PHASE 2: Optimized Retrieval Workflow ---")
    query = "How many vacation days do I get?"
    print(f"[USER QUERY]: {query}\n")
    
    print("[SYSTEM]: 1. Fetching Top 10 broad matches from Pinecone...")
    print("[SYSTEM]: 2. Sending 10 matches to Cohere for deep re-ranking...")
    print("[SYSTEM]: 3. Cohere returning Top 3 highest-confidence chunks.\n")

    try:
        # This will trigger the two-step Recall -> Rerank pipeline
        compressed_docs = retriever.invoke(query)
        
        print(">>> FINAL PRUNED CONTEXT SENT TO LLM <<<")
        for i, doc in enumerate(compressed_docs):
            # Cohere attaches the relevance score to the metadata
            score = doc.metadata.get("relevance_score", "N/A")
            print(f"Result {i+1} [Rerank Score: {score}]: {doc.page_content}")
            
    except Exception as e:
        print(f"[ERROR during retrieval]: API limits or connection failed: {e}")

if __name__ == "__main__":
    run_demo()
