"""
Demo: Document Loaders, Splitting, and Custom Retrievers
This script simulates the foundational RAG data pipeline:
1. Loading unstructured data (a giant string).
2. Intelligently splitting it so sentences aren't cut in half.
3. Creating a Retriever wrapper to query the underlying "vector DB".
"""

# =====================================================================
# 1. Simulating LangChain Document Loading
# =====================================================================
class Document:
    """Mock LangChain Document object containing page content and metadata."""
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content # Unstructured text
        self.metadata = metadata or {} # Structured JSON dict for filtering

def load_simulated_financial_report() -> Document:
    print("--- 1. Document Loading ---")
    print("Loading 'Q3_Financial_Summary.pdf' into memory...")
    
    # Imagine a 50-page PDF smashed into one giant continuous string string after being scraped.
    raw_text = (
        "Q3 Financial Summary Report.\n"
        "Executive Overview: Revenue grew by 15% due to enterprise saas sales. "
        "Costs were significantly reduced in Q3. We saved $5M on cloud hosting alone. "
        "Looking Ahead: Q4 projections indicate a slight dip in consumer software sales, "
        "but enterprise contracts will carry the deficit. "
        "Risks: Supply chain hardware shortages may delay the new server rollout."
    )
    
    # Packaged neatly into a standard Document object format for downstream pipelines
    return Document(page_content=raw_text, metadata={"source": "Q3_Summary.pdf"})

# =====================================================================
# 2. Text Splitting (Chunking)
# =====================================================================
def simulate_recursive_text_splitter(doc: Document, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """
    Simulates LangChain's RecursiveCharacterTextSplitter.
    It breaks a massive string into smaller bites, but with an overlap
    so context is preserved at the boundaries.
    """
    print(f"\n--- 2. Text Splitting (Chunking) ---")
    print(f"Applying Size={chunk_size}, Overlap={chunk_overlap}")
    
    # Simple split by spaces roughly simulating token-based or character-based chunking
    words = doc.page_content.split()
    chunks = []
    
    # Simple simulated logic walking through words using size/overlap logic
    i = 0
    while i < len(words):
        # Grab a chunk of words up to the chunk_size
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        # Create a new Document fragment carrying the previous metadata forward PLUS a chunk_id
        chunk_doc = Document(
            page_content=chunk_text, 
            metadata={**doc.metadata, "chunk_id": len(chunks)}
        )
        chunks.append(chunk_doc)
        
        # Advance the pointer, minus the overlap, to create redundancy so sentences aren't cleanly sliced in half
        i += (chunk_size - chunk_overlap)
        
    print(f"Generated {len(chunks)} overlapping chunks.")
    return chunks

# =====================================================================
# 3. Custom Retriever Interface
# =====================================================================
class SimulatedRetriever:
    """
    Simulates `vectorstore.as_retriever()`. It wraps the chunks (the database) and provides
    a single `.invoke()` method that magically returns the best fitting chunks.
    """
    def __init__(self, chunks: list[Document]):
        # Normally this is a connection pool to a DB, but here we just store the List
        self.chunks = chunks
        print("\n--- 3. Base Retriever Created ---")
        print("Vector database wrapper fully initialized.")
        
    def invoke(self, query: str, top_k: int = 2) -> list[Document]:
        """Simulates finding the most semantically relevant chunks."""
        print(f"\n[Retriever] Executing search for: '{query}'")
        
        # Simple simulated similarity logic acting as an embedding + cosine similarity match
        query = query.lower()
        results = []
        for chunk in self.chunks:
            score = 0
            # If asking about saving costs
            if "cost" in query or "save" in query:
                if "costs" in chunk.page_content.lower() or "saved" in chunk.page_content.lower():
                    score += 10
            # If asking about future projections
            if "q4" in query or "future" in query:
                if "q4" in chunk.page_content.lower() or "ahead" in chunk.page_content.lower():
                    score += 10
                    
            if score > 0:
                results.append((score, chunk))
                
        # Sort and return top K Document objects based on their simulated relevance score
        results.sort(key=lambda x: x[0], reverse=True)
        return [res[1] for res in results][:top_k]

# =====================================================================
# 4. Pipeline Execution
# =====================================================================
def run_retriever_pipeline():
    print("=== Agentic AI Fundamentals: Document Loaders & Retrievers ===")
    
    # 1. Load Data
    master_document = load_simulated_financial_report()
    
    # 2. Chunk Data (Size=12 words, Overlap=3 words for demo purposes)
    doc_chunks = simulate_recursive_text_splitter(master_document, chunk_size=12, chunk_overlap=3)
    
    print("\n[Debug] Inspecting Chunk Overlap:")
    for c in doc_chunks:
        print(f"  Chunk {c.metadata['chunk_id']}: {c.page_content}")
        
    # 3. Create Retriever
    retriever = SimulatedRetriever(doc_chunks)
    
    # 4. Execute query - The retriever interface abstracts away the math and just gives us back Documents
    user_query = "Did we save any money on costs recently?"
    retrieved_docs = retriever.invoke(user_query, top_k=1)
    
    print("\n>>> RETRIEVED CONTEXT <<<")
    for doc in retrieved_docs:
        print(f"Source: {doc.metadata['source']} (Chunk {doc.metadata['chunk_id']})")
        print(f"Content: '{doc.page_content}'")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_retriever_pipeline()
