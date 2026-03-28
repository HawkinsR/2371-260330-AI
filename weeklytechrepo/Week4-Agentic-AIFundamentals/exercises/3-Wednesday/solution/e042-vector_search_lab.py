from typing import List, Dict, Any
import math

# =====================================================================
# MOCK HELPER FUNCTIONS (Do not edit this section)
# =====================================================================
def get_mock_embedding(text: str) -> List[float]:
    text = text.lower()
    if "sick" in text or "medical" in text or "doctor" in text:
        return [0.8, 0.1, 0.1]
    elif "password" in text or "vpn" in text or "reset" in text:
        return [0.1, 0.8, 0.1]
    elif "paycheck" in text or "deposit" in text:
        return [0.1, 0.1, 0.8]
    return [0.5, 0.5, 0.5]

def calculate_cosine_similarity(vecA: List[float], vecB: List[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vecA, vecB))
    magnitude_a = math.sqrt(sum(a * a for a in vecA))
    magnitude_b = math.sqrt(sum(b * b for b in vecB))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)

class MockPineconeIndex:
    def __init__(self, name: str):
        self.name = name
        self.vectors: Dict[str, Dict[str, Any]] = {}

    def upsert(self, records: List[Dict[str, Any]], namespace: str = "default"):
        print(f"\n[Pinecone] Upserting records into namespace '{namespace}'...")
        for rec in records:
            self.vectors[rec["id"]] = {
                "vector": rec["values"],
                "metadata": rec.get("metadata", {}),
                "namespace": namespace
            }

    def query(self, vector: List[float], top_k: int, namespace: str, filter_dict: Dict = None) -> List[Dict]:
        results = []
        for vid, data in self.vectors.items():
            if data["namespace"] != namespace:
                continue
            
            if filter_dict:
                match = True
                for key, condition in filter_dict.items():
                    if "$eq" in condition and data["metadata"].get(key) != condition["$eq"]:
                        match = False
                if not match:
                    continue
                    
            score = calculate_cosine_similarity(vector, data["vector"])
            results.append({"id": vid, "score": score, "metadata": data["metadata"]})
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

# =====================================================================
# YOUR TASKS
# =====================================================================
def ingest_knowledge_base(db: MockPineconeIndex, doc_id: str, doc_text: str):
    print("--- 1. Ingesting Document into Vector DB ---")
    
    # 1. Convert the string doc_text into a math vector
    doc_vector = get_mock_embedding(doc_text)
    
    # 2. Construct the record dictionary with id, values, and metadata (must include "text": doc_text)
    record = {
        "id": doc_id,
        "values": doc_vector,
        "metadata": {"text": doc_text}
    }
    
    # 3. Call .upsert() on the db. Pass the record in a list. Set namespace to "hr-policies"
    db.upsert([record], namespace="hr-policies")
    print("✓ Upsert Complete")

def retrieve_answer(db: MockPineconeIndex, user_question: str) -> str:
    print("\n--- 2. Semantic Vector Search Search ---")
    print(f"User Query: '{user_question}'")
    
    # 4. Convert the question string into a query vector
    query_vector = get_mock_embedding(user_question)
    
    # 5. Call .query() on the db. Request top_k=1 against the "hr-policies" namespace
    results = db.query(
        vector=query_vector,
        top_k=1,
        namespace="hr-policies"
    )
    
    if results and len(results) > 0:
        match = results[0]
        print(f"Match Score: {match['score']:.4f}")
        return match["metadata"].get("text", "No text found in metadata")
    
    return "No match found."

if __name__ == "__main__":
    print("=== Agentic AI: Knowledge Base Vectors ===")
    
    # Initialize the "Cloud" DB
    pinecone_db = MockPineconeIndex("corporate-docs")
    
    # The raw document
    hr_doc = "Medical Policy 401A: If you require a doctor visit or feel violently ill, notify your manager and take sick leave."
    
    # Execute Ingestion
    ingest_knowledge_base(pinecone_db, "doc_hr_01", hr_doc)
    
    # A user asks a question with totally different vocabulary
    query = "I am feeling sick today, who do I call?"
    
    # Execute Retrieval
    answer = retrieve_answer(pinecone_db, query)
    
    print("\n>>> RETRIEVED KNOWLEDGE <<<")
    print(answer)
    print("===========================\n")
