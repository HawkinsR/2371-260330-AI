# =====================================================================
# MOCK LANGCHAIN CLASSES (Do not edit this section)
# =====================================================================
class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class SimulatedRetriever:
    def __init__(self, chunks: list[Document]):
        self.chunks = chunks
        
    def invoke(self, query: str, top_k: int = 1) -> list[Document]:
        print(f"\n[Retriever] Executing search for: '{query}'")
        query = query.lower()
        results = []
        
        for chunk in self.chunks:
            score = 0
            if "hardware" in query or "delay" in query:
                if "hardware" in chunk.page_content.lower() or "delay" in chunk.page_content.lower():
                    score += 10
            if score > 0:
                results.append((score, chunk))
                
        results.sort(key=lambda x: x[0], reverse=True)
        return [res[1] for res in results][:top_k]

# =====================================================================
# YOUR TASKS
# =====================================================================
def load_raw_data() -> str:
    return (
        "Welcome to the company. "
        "Your health benefits kick in after 30 days. "
        "Please note there is a hardware delay "
        "affecting new laptop shipments this month."
    )

def chunk_document(master_text: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    print(f"\n--- 1. Text Splitting (Size={chunk_size}, Overlap={chunk_overlap}) ---")
    
    # 1. Split the master_text into a list of words
    words = master_text.split()
    chunks = []
    
    # 2. Iterate over the words, extracting chunks of size 'chunk_size'
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        chunk_doc = Document(
            page_content=chunk_text, 
            metadata={"chunk_id": len(chunks)}
        )
        chunks.append(chunk_doc)
        
        # Advance your pointer by (chunk_size - chunk_overlap)
        i += (chunk_size - chunk_overlap)
        
    return chunks

# =====================================================================
# 4. Pipeline Execution
# =====================================================================
if __name__ == "__main__":
    print("=== Agentic AI: Document Chunking Pipeline ===")
    
    raw_text = load_raw_data()
    
    # We want 8 words per chunk, with 2 words of overlap
    doc_chunks = chunk_document(raw_text, chunk_size=8, chunk_overlap=2)
    
    if not doc_chunks:
        print("ERROR: Chunks list is empty. Complete the chunk_document function.")
    else:
        print("\n[Debug] Inspecting Generated Chunks:")
        for c in doc_chunks:
            print(f"  Chunk {c.metadata.get('chunk_id')}: '{c.page_content}'")
            
        # Create Retriever
        retriever = SimulatedRetriever(doc_chunks)
        
        # Execute query
        user_query = "Is there a hardware delay right now?"
        retrieved_docs = retriever.invoke(user_query, top_k=1)
        
        print("\n>>> RETRIEVED CONTEXT <<<")
        for doc in retrieved_docs:
            print(f"Source Chunk: {doc.metadata.get('chunk_id')}")
            print(f"Content: '{doc.page_content}'")
        print("="*50 + "\n")
