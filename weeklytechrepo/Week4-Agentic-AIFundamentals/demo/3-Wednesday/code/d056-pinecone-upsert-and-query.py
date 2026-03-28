"""
Demo: Vector Databases, Embeddings, and Pinecone Search
This script simulates how text is converted into multi-dimensional vectors by an 
Embedding Model, stored in a cloud Vector Database (Pinecone), and retrieved 
using Cosine Similarity combined with hard Metadata Filtering.
"""

import math
from typing import List, Dict, Any

# =====================================================================
# 1. Simulating an Embedding Model (Text -> Numbers)
# =====================================================================
def get_mock_embedding(text: str) -> List[float]:
    """
    Simulates OpenAI's text-embedding-3-small (which returns 1536 floats).
    For this demo, we return a tiny 3-dimensional vector so humans can read it.
    """
    text = text.lower()
    
    # Financial concepts map to high values on the X-axis
    # Tech concepts map to high values on the Y-axis
    # HR concepts map to high values on the Z-axis
    
    if "expense" in text or "reimburse" in text:
        return [0.9, 0.1, 0.0]
    elif "password" in text or "vpn" in text:
        return [0.0, 0.9, 0.1]
    elif "pto" in text or "vacation" in text:
        return [0.1, 0.0, 0.9]
    else:
        # Default neutral vector if no keywords match
        return [0.5, 0.5, 0.5]

def calculate_cosine_similarity(vecA: List[float], vecB: List[float]) -> float:
    """Calculates the geometric angle between two vectors (Cosine Similarity)."""
    # Sum of the products of corresponding elements
    dot_product = sum(a * b for a, b in zip(vecA, vecB))
    # Length of vector A
    magnitude_a = math.sqrt(sum(a * a for a in vecA))
    # Length of vector B
    magnitude_b = math.sqrt(sum(b * b for b in vecB))
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
        
    # Cosine Similarity is Dot Product divided by product of magnitudes
    return dot_product / (magnitude_a * magnitude_b)

# =====================================================================
# 2. Simulating the Pinecone Database
# =====================================================================
class MockPineconeIndex:
    """Simulates a Pinecone cloud vector database."""
    def __init__(self, name: str):
        self.name = name
        # A dictionary acting as our raw database table
        self.vectors: Dict[str, Dict[str, Any]] = {}

    def upsert(self, records: List[Dict[str, Any]], namespace: str = "default"):
        """Inserts or updates records into the vector database."""
        print(f"\n[Pinecone] Upserting {len(records)} records into namespace '{namespace}'...")
        for rec in records:
            # Store the vector and its metadata tied to a unique ID
            self.vectors[rec["id"]] = {
                "vector": rec["values"],
                "metadata": rec.get("metadata", {}),
                "namespace": namespace # Think of a namespace like a specific folder inside the DB
            }
            print(f"  -> Saved {rec['id']} | Vector: {rec['values']} | Meta: {rec.get('metadata')}")

    def query(self, vector: List[float], top_k: int, namespace: str, filter_dict: Dict = None) -> List[Dict]:
        """Queries the database to find the K closest vectors to our target."""
        print(f"\n[Pinecone] Performing Vector Search...")
        print(f"  -> Target Vector: {vector}")
        print(f"  -> Filters Applied: namespace='{namespace}', metadata={filter_dict}")
        
        results = []
        for vid, data in self.vectors.items():
            # Apply Namespace filter (only search inside this specific "folder")
            if data["namespace"] != namespace:
                continue
                
            # Apply Metadata filter (Simulating the MongoDB-style $eq operator for hard filtering)
            if filter_dict:
                match = True
                for key, condition in filter_dict.items():
                    # If the condition dictates it MUST equal this value, and it doesn't, skip it
                    if "$eq" in condition and data["metadata"].get(key) != condition["$eq"]:
                        match = False
                if not match:
                    continue
            
            # Mathematical similarity phase: how close is our query to the stored vector?
            score = calculate_cosine_similarity(vector, data["vector"])
            results.append({"id": vid, "score": score, "metadata": data["metadata"]})
            
        # Sort by highest score first and grab Top K results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

# =====================================================================
# 3. Execution Pipeline
# =====================================================================
def run_vector_demo():
    print("=== Agentic AI Fundamentals: Vector Search & Embeddings ===")
    
    # Initialize the "Cloud" DB index
    db = MockPineconeIndex("corporate-docs")
    
    # --- PHASE 1: Data Ingestion (Upsert) ---
    doc1_text = "How to file a travel expense report."
    doc2_text = "How to reset your corporate VPN password."
    doc3_text = "Approving PTO and vacation days in Workday."
    
    # We turn human text into mathematical vectors via the "Embedding Model"
    v1 = get_mock_embedding(doc1_text)
    v2 = get_mock_embedding(doc2_text)
    v3 = get_mock_embedding(doc3_text)
    
    # We save the math + JSON metadata to the cloud in a specific format
    docs_to_upsert = [
        {"id": "doc1_finance", "values": v1, "metadata": {"dept": "finance", "year": 2024, "text": doc1_text}},
        {"id": "doc2_it",      "values": v2, "metadata": {"dept": "it",      "year": 2024, "text": doc2_text}},
        {"id": "doc3_hr",      "values": v3, "metadata": {"dept": "hr",      "year": 2024, "text": doc3_text}}
    ]
    # Uploading them to a specific partition named "employee-handbook-v2"
    db.upsert(docs_to_upsert, namespace="employee-handbook-v2")
    
    # --- PHASE 2: Retrieval (Query) ---
    print("\n" + "-"*50)
    user_query = "I need to get reimbursed for my flight."
    print(f"[User Asks]: '{user_query}'")
    
    # Instead of doing a SQL 'SELECT * WHERE text LIKE %flight%',
    # We convert the question into the identical mathematical space
    query_vec = get_mock_embedding(user_query)
    
    # We ask Pinecone to find the vectors closest to our query vector
    # We ALSO pass a hard filter. Even if a document is similar, reject it if it isn't 'finance'
    search_results = db.query(
        vector=query_vec,
        top_k=2, # Give me the top 2 best matches
        namespace="employee-handbook-v2", # Look only in this bucket
        filter_dict={"dept": {"$eq": "finance"}} # Only accept documents flagged for finance
    )
    
    print("\n>>> SEARCH RESULTS RETRIEVED <<<")
    for res in search_results:
        # The result returns the ID, the Cosine Score, and the attached Metadata so we can fetch the document
        print(f"Match: {res['id']}")
        print(f"Similarity Score: {res['score']:.4f}")
        print(f"Content String: '{res['metadata']['text']}'\n")

    print("="*50 + "\n")

if __name__ == "__main__":
    run_vector_demo()
