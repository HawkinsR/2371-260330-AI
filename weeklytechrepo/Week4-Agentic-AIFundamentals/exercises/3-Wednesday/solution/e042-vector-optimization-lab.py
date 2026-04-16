from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# =====================================================================
# 1. Initialize Embeddings
# =====================================================================
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1"
)

# =====================================================================
# 2. Create the Vector Store
# =====================================================================
# NOTE: Ensure your Pinecone index dimension matches Titan v2 (256 dims)
vectorstore = PineconeVectorStore(
    index_name="corp-knowledge-v4",
    embedding=embeddings,
    namespace="hr-policies"
)

# =====================================================================
# 3. Add Sample Documents (Run Only Once)
# =====================================================================
sample_docs = [
    Document(page_content="Full-time employees receive 20 days of Paid Time Off per year.", metadata={"topic": "pto"}),
    Document(page_content="PTO must be requested 2 weeks in advance through the HR portal.", metadata={"topic": "pto"}),
    Document(page_content="Unused PTO of up to 5 days may be rolled over to the next calendar year.", metadata={"topic": "pto"}),
    Document(page_content="The company 401k match is 4% with a 2-year vesting schedule.", metadata={"topic": "benefits"}),
    Document(page_content="Remote work is permitted up to 3 days per week with manager approval.", metadata={"topic": "remote"}),
]
# vectorstore.add_documents(sample_docs) # Run once to populate the index

# =====================================================================
# 4. Build the Two-Stage Retrieval Pipeline
# =====================================================================
# Stage 1: Broad recall from Pinecone (fast, approximate)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Stage 2: Precision re-ranking with Cohere (slow, accurate)
compressor = CohereRerank(model="rerank-english-v3.0", top_n=2)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# =====================================================================
# 5. Query and Display
# =====================================================================
def run_exercise():
    print("=== e042: Optimizing Vector Retrieval ===\n")
    query = "How many vacation days do I get per year?"
    print(f"Query: {query}\n")
    
    results = retriever.invoke(query)
    
    print(f"[RESULTS]: Cohere selected top {len(results)} from 5 Pinecone candidates:\n")
    for i, doc in enumerate(results):
        score = doc.metadata.get("relevance_score", "N/A")
        print(f"  Result {i+1} [Cohere Score: {score:.2f}]: {doc.page_content}")

if __name__ == "__main__":
    run_exercise()
