"""
Demo: Multi-Query and Hybrid Search
This script demonstrates how to split a complex query into simple sub-queries,
search across two simulated indices (Dense and Sparse), deduplicate results, 
and re-rank them for the final context window.
"""

from typing import List, Dict

# =====================================================================
# SIMULATED DATABASES AND MODELS
# =====================================================================
# A mocked internal knowledge base for demonstration
DUMMY_CORPUS = {
    "doc1": "2024 healthcare policy includes vision and dental.",
    "doc2": "The 2022 policy did not have vision benefits.",
    "doc3": "Error code ERR-404-B means the server connection dropped.",
    "doc4": "Employees get 15 days PTO in 2024.",
    "doc5": "In 2022, employees had 12 days PTO."
}

def simulate_llm_query_decomposition(query: str) -> List[str]:
    """Takes a complex human question and splits it into searchable fragments."""
    print(f"\n[Multi-Query Router] Deconstructing complex query: '{query}'")
    
    # Simulate an LLM breaking down the question into distinct vectors
    if "PTO" in query and "2024" in query and "2022" in query:
        sub_queries = [
            "What is the PTO policy for 2024?",
            "What was the PTO policy in 2022?",
            "Comparison of time off benefits between 2022 and 2024"
        ]
    else:
        # Fallback if the query is already simple enough
        sub_queries = [query]
        
    for i, sq in enumerate(sub_queries):
        print(f"  -> Generated Sub-query {i+1}: '{sq}'")
    return sub_queries

def simulate_dense_search(query: str) -> List[str]:
    """Simulates a Vector Embeddings Database focusing on Semantic Meaning (ideas, not words)."""
    # Dense search is good at semantic meaning 
    if "2024" in query and "PTO" in query:
        return ["doc4", "doc1"]
    elif "2022" in query and "PTO" in query:
        return ["doc5", "doc2"]
    return ["doc1", "doc2"]

def simulate_sparse_search(query: str) -> List[str]:
    """Simulates a BM25 Database focusing on exact Keyword Matches."""
    # Sparse (BM25) is exact keyword matching
    if "PTO" in query:
        return ["doc4", "doc5"]
    return []

def simulate_cross_encoder_reranker(query: str, docs: List[str]) -> List[str]:
    """Takes a raw list of documents and re-orders them based on relevance to the original prompt."""
    print("\n[Re-ranker] Scoring extracted documents against original query...")
    
    # Simulate scoring algorithm placing most relevant docs first
    scored_docs = []
    for doc_id in docs:
        content = DUMMY_CORPUS[doc_id]
        score = 0
        
        # Simple simulated scoring rubric depending on presence of keywords
        if "PTO" in content: score += 5
        if "2024" in content or "2022" in content: score += 2
            
        scored_docs.append((score, content))
        print(f"  -> Scored [{doc_id}]: {score}/10")
        
    # Sort descending based on the score we just calculated
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    
    # Return the content strings of the Top 2 documents
    top_docs = [content for score, content in scored_docs[:2]]
    return top_docs

# =====================================================================
# THE PIPELINE
# =====================================================================
def demonstrate_advanced_retrieval():
    print("--- Advanced Retrieval Pipeline Demo ---")
    
    # The starting point: A complex user question that would confuse a single vector search
    complex_query = "Compare the PTO policies from 2022 to the 2024 updates."
    print(f"USER ASK: {complex_query}")
    
    # 1. Multi-Query Decomposition: Turn one complex question into three simple ones
    sub_queries = simulate_llm_query_decomposition(complex_query)
    
    # 2. Hybrid Search Configuration (Sparse + Dense)
    # Using a Python set() automatically ensures we don't return duplicates if both DBs find the same doc
    all_retrieved_docs = set() 
    
    print("\n[Hybrid Search] Executing against Vector and BM25 databases...")
    for sq in sub_queries:
        # Ask the meaning-based database
        dense_results = simulate_dense_search(sq)
        # Ask the exact-keyword-based database
        sparse_results = simulate_sparse_search(sq)
        
        # Add all unique IDs to the master list
        all_retrieved_docs.update(dense_results)
        all_retrieved_docs.update(sparse_results)
        
    print(f"-> Unsorted, deduplicated raw results: {list(all_retrieved_docs)}")
    
    # 3. Re-ranking
    # We pass the ORIGINAL complex query and the raw docs to the Cross-Encoder to order them properly
    top_context = simulate_cross_encoder_reranker(complex_query, list(all_retrieved_docs))
    
    # 4. Final Aggregation
    print("\n" + "="*50)
    print(">>> FINAL CONTEXT INJECTED INTO MAIN LLM <<<")
    # This is what gets pasted invisibly into the LLM context window right before it speaks
    for i, ctx in enumerate(top_context):
        print(f"Context {i+1}: {ctx}")
    print("="*50 + "\n")

if __name__ == "__main__":
    demonstrate_advanced_retrieval()
