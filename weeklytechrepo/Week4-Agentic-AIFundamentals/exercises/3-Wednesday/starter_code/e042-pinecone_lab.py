import time
from typing import List, Dict, Any

# =====================================================================
# 1. Pinecone Serverless Logic (Mock)
# =====================================================================
class MockPineconeServerless:
    """
    Simulates the serverless architecture of Pinecone.
    """
    def __init__(self):
        # Storage format: {namespace: {id: {values: [...], metadata: {...}}}}
        self.indexes: Dict[str, Dict[str, Any]] = {}

    def upsert(self, records: List[Dict], namespace: str):
        # 1. TODO: Implement serverless upsert.
        # Initialize namespace if not exists.
        # Store each record by its 'id'.
        print(f"   [PINECONE]: Upserting {len(records)} vectors to '{namespace}'...")
        pass

    def query(self, vector: List[float], top_k: int, namespace: str, filters: Dict = None) -> List[Dict]:
        # 2. TODO: Implement basic query logic (return dummy results for now).
        # In a real setup, this would calculate cosine similarity.
        print(f"   [PINECONE]: Querying namespace '{namespace}'...")
        return []

# =====================================================================
# 2. LangSmith Observability
# =====================================================================
def log_to_langsmith(run_name: str, inputs: Dict, outputs: Dict, start_time: float):
    """
    Simulates sending a trace to the LangSmith dashboard.
    """
    # 3. TODO: Calculate latency using 'time.time() - start_time'.
    latency = 0.0
    
    print(f"\n   [LANGSMITH TRACE]: {run_name}")
    print(f"   - Inputs: {list(inputs.keys())}")
    print(f"   - Latency: {latency:.4f}s")
    print(f"   - Status: SUCCESS")

# =====================================================================
# 3. Execution Pipeline
# =====================================================================
def run_vector_pipeline(user_query: str, dept: str):
    db = MockPineconeServerless()
    
    # --- Step 1: Upsert Knowledge ---
    print(f"--- Action: Upserting {dept} Documents ---")
    start = time.time()
    doc_id = "doc_001"
    doc_meta = {"text": "Vacation policy is 15 days.", "dept": dept}
    
    # 4. TODO: Call db.upsert() with a properly formatted record.
    # Record format: {"id": doc_id, "values": [0.1, 0.2], "metadata": doc_meta}
    
    log_to_langsmith("Pinecone_Upsert", {"id": doc_id}, {"status": "ok"}, start)

    # --- Step 2: Query Knowledge ---
    print(f"\n--- Action: Querying {dept} Knowledge ---")
    start = time.time()
    
    # 5. TODO: Call db.query() with a filter for the specific department.
    # filters = {"dept": {"$eq": dept}}
    results = []
    
    log_to_langsmith("Pinecone_Query", {"query": user_query}, {"matches": len(results)}, start)

if __name__ == "__main__":
    print("=== Week 4: Pinecone Serverless & LangSmith Lab ===")
    run_vector_pipeline("What is the vacation policy?", "HR")
