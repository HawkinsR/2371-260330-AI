from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# =====================================================================
# 1. Initialize Embeddings
# =====================================================================
# TODO: Create a BedrockEmbeddings instance.
# Use model_id="amazon.titan-embed-text-v2:0"

# embeddings = BedrockEmbeddings(...)

# =====================================================================
# 2. Create the Vector Store
# =====================================================================
# TODO: Use PineconeVectorStore to connect to your Pinecone index.
# Use index_name="your-index-name" and the embeddings from above.
# PINECONE_API_KEY must be in your environment.

# vectorstore = PineconeVectorStore(index_name=..., embedding=...)

# =====================================================================
# 3. Add Sample Documents (Run Only Once!)
# =====================================================================
sample_docs = [
    Document(page_content="Full-time employees receive 20 days of Paid Time Off per year.", metadata={"topic": "pto"}),
    Document(page_content="PTO must be requested 2 weeks in advance through the HR portal.", metadata={"topic": "pto"}),
    Document(page_content="Unused PTO of up to 5 days may be rolled over to the next calendar year.", metadata={"topic": "pto"}),
    Document(page_content="The company 401k match is 4% with a 2-year vesting schedule.", metadata={"topic": "benefits"}),
    Document(page_content="Remote work is permitted up to 3 days per week with manager approval.", metadata={"topic": "remote"}),
]

# TODO: Uncomment to upsert documents into Pinecone (run once only)
# vectorstore.add_documents(sample_docs)

# =====================================================================
# 4. Build the Two-Stage Retrieval Pipeline
# =====================================================================
# TODO: Create a base retriever from the vectorstore using .as_retriever().
# Use search_kwargs={"k": 5} to recall the top 5 candidates.

# base_retriever = vectorstore.as_retriever(...)

# TODO: Create a CohereRerank compressor with top_n=2 to return top 2 results.
# Wrap it in a ContextualCompressionRetriever.

# compressor = CohereRerank(...)
# retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

# =====================================================================
# 5. Query and Display
# =====================================================================
def run_exercise():
    print("=== e042: Optimizing Vector Retrieval ===\n")
    query = "How many vacation days do I get per year?"
    
    # TODO: Invoke your retriever with the query above.
    # Print each result's relevance_score (from metadata) and page_content.
    print(f"Query: {query}\n")
    # YOUR CODE HERE

if __name__ == "__main__":
    run_exercise()
