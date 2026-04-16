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
        if namespace not in self.indexes:
            self.indexes[namespace] = {}
        
        print(f"   [PINECONE]: Upserting {len(records)} vectors to '{namespace}'...")
        for rec in records:
            self.indexes[namespace][rec["id"]] = {
                "values": rec["values"],
                "metadata": rec["metadata"]
            }

    def query(self, vector: List[float], top_k: int, namespace: str, filters: Dict = None) -> List[Dict]:
        print(f"   [PINECONE]: Querying namespace '{namespace}' with filters: {filters}...")
        
        if namespace not in self.indexes:
            return []
            
        results = []
        for vid, data in self.indexes[namespace].items():
            # Minimal filter logic simulator
            if filters:
                match = True
                for key, condition in filters.items():
                    if "$eq" in condition and data["metadata"].get(key) != condition["$eq"]:
                        match = False
                if not match:
                    continue
            
            # Since this is a mock, we return a dummy score and the metadata
            results.append({"id": vid, "score": 0.95, "metadata": data["metadata"]})
            
        return results[:top_k]

# =====================================================================
# 2. LangSmith Observability
# =====================================================================
def log_to_langsmith(run_name: str, inputs: Dict, outputs: Dict, start_time: float):
    """
    Simulates sending a trace to the LangSmith dashboard.
    """
    latency = time.time() - start_time
    
    print(f"\n   [LANGSMITH TRACE]: {run_name}")
    print(f"   - Inputs: {list(inputs.keys())}")
    print(f"   - Outputs: {list(outputs.keys())}")
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
    
    # Task: Wrap records in a list and call upsert
    db.upsert([
        {"id": doc_id, "values": [0.1, 0.2, 0.3], "metadata": doc_meta}
    ], namespace="corporate-knowledge")
    
    log_to_langsmith("Pinecone_Upsert", {"id": doc_id}, {"status": "ok"}, start)

    # --- Step 2: Query Knowledge ---
    print(f"\n--- Action: Querying {dept} Knowledge ---")
    start = time.time()
    
    # Task: Perform a hybrid search with metadata filtering
    results = db.query(
        vector=[0.1, 0.2, 0.3], 
        top_k=1, 
        namespace="corporate-knowledge",
        filters={"dept": {"$eq": dept}}
    )
    
    log_to_langsmith("Pinecone_Query", {"query": user_query}, {"matches": len(results)}, start)
    
    if results:
        print(f"\nFinal Result: {results[0]['metadata']['text']}")

if __name__ == "__main__":
    print("=== Week 4: Pinecone Serverless & LangSmith Lab ===")
    run_vector_pipeline("What is the vacation policy?", "HR")
