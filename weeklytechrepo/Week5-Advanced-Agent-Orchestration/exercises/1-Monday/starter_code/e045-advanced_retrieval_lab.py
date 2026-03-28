from typing import List

# =====================================================================
# MOCK LLM & DATABASE (Do not edit this section)
# =====================================================================
DUMMY_CORPUS = {
    "doc1": "Remote work policy: Employees may work from home permanently.",
    "doc2": "Hybrid policy: Employees must be in the office 2 days a week.",
    "doc3": "Vacation policy: 15 days of PTO per year.",
    "doc4": "IT Policy: Using a VPN is mandatory for remote connections."
}

def simulate_dense_search(query: str) -> List[str]:
    query = query.lower()
    if "remote" in query: return ["doc1", "doc4"]
    if "hybrid" in query: return ["doc2", "doc3"]
    return ["doc1", "doc2"]

def simulate_sparse_search(query: str) -> List[str]:
    query = query.lower()
    if "policy" in query: return ["doc1", "doc2", "doc3", "doc4"]
    return []

# =====================================================================
# YOUR TASKS
# =====================================================================

def decompose_query(complex_query: str) -> List[str]:
    print(f"\n[Router] Decomposing query: '{complex_query}'")
    
    # 1. TODO: If complex_query contains BOTH "remote" and "hybrid"
    # return a list of two simpler queries. Otherwise, return the original.
    sub_queries = []
    
    for i, sq in enumerate(sub_queries):
         print(f"  -> Generated Sub-query {i+1}: '{sq}'")
    return sub_queries


def execute_hybrid_search(sub_queries: List[str]) -> List[str]:
    print("\n[Hybrid Search] Executing against Vector and BM25 databases...")
    all_retrieved_docs = set()
    
    # 2. TODO: Iterate over sub_queries. 
    # Call simulate_dense_search() and simulate_sparse_search() for each.
    # Add the returned doc IDs to the all_retrieved_docs set.
    
    
    print(f"  -> Deduplicated raw results: {list(all_retrieved_docs)}")
    return list(all_retrieved_docs)


def rerank_documents(original_query: str, doc_ids: List[str], top_k: int = 2) -> List[str]:
    print("\n[Re-ranker] Scoring extracted documents...")
    scored_docs = []
    
    # 3. TODO: Iterate over doc_ids. Fetch content from DUMMY_CORPUS.
    # Score the content based on the presence of keywords: "remote" (+5), "hybrid" (+5), "policy" (+2)
    # Store tuples of (score, content) in scored_docs.
    
    
    # 4. TODO: Sort scored_docs descending by score, and isolate the top_k text contents into top_docs.
    top_docs = []
    
    for score, content in scored_docs[:top_k]:
        print(f"  -> Scored [{score}/12]: {content}")
        
    return top_docs

# =====================================================================
# PIPELINE EXECUTION
# =====================================================================
def run_pipeline():
    print("=== Agentic AI: Advanced Retrieval Pipeline ===")
    
    user_query = "Compare the remote work policy to the hybrid work policy."
    
    # 1. Decompose
    queries = decompose_query(user_query)
    
    if not queries:
        print("ERROR: decompose_query returned empty. Finish the function.")
        return
        
    # 2. Hybrid Retrieve
    raw_docs = execute_hybrid_search(queries)
    
    # 3. Re-Rank
    final_context = rerank_documents(user_query, raw_docs, top_k=2)
    
    print("\n" + "="*50)
    print(">>> FINAL CONTEXT FOR LLM <<<")
    for ctx in final_context:
        print(f"- {ctx}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_pipeline()
