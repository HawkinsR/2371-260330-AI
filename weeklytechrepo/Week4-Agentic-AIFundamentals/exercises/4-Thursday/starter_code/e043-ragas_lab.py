import time
from typing import List, Dict

# =====================================================================
# 1. RAGAS Evaluation Logic (Mock)
# =====================================================================
class MockRagasEvaluator:
    """
    Simulates an LLM-as-a-judge for RAG evaluation.
    """
    def evaluate(self, query: str, context: str, answer: str) -> Dict[str, float]:
        print(f"\n[RAGAS]: Evaluating response for query: '{query}'...")
        time.sleep(1) # Simulate LLM analysis time
        
        # 1. TODO: Implement Faithfulness scoring.
        # Check if key facts in the 'answer' are present in the 'context'.
        # If context says '15 days' but answer says 'unlimited', faithfulness should be 0.0.
        faithfulness = 1.0
        
        # 2. TODO: Implement Answer Relevancy scoring.
        # Check if the 'answer' actually addresses the 'query'.
        relevancy = 1.0
        
        return {
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy
        }

# =====================================================================
# 2. Execution Pipeline
# =====================================================================
def run_evaluation_pipeline(user_query: str, retrieved_context: str, generated_answer: str):
    evaluator = MockRagasEvaluator()
    
    # --- Action: Run Evaluation ---
    metrics = evaluator.evaluate(user_query, retrieved_context, generated_answer)
    
    print("\n>>> RAGAS METRICS <<<")
    for metric, score in metrics.items():
        # 3. TODO: Log a 'CRITICAL' alert if faithfulness score is below 0.5.
        status = "PASS" if score >= 0.7 else "FAIL"
        print(f"   - {metric.capitalize()}: {score:.2f} [{status}]")

if __name__ == "__main__":
    print("=== Week 4: Retrieval Evaluation (RAGAS) Lab ===")
    
    # Test Case 1: Grounded Answer (Faithful)
    print("\n--- Test Case 1: Grounded Answer ---")
    q1 = "How many vacation days do I get?"
    c1 = "Employees are entitled to 15 days of paid vacation per year."
    a1 = "You get 15 days of paid vacation annually."
    run_evaluation_pipeline(q1, c1, a1)

    # Test Case 2: Hallucinated Answer (Unfaithful)
    print("\n--- Test Case 2: Hallucinated Answer ---")
    q2 = "What is the vacation policy?"
    c2 = "Standard vacation is 15 days per year."
    # 4. TODO: Change 'a2' to something that conflicts with 'c2' to trigger a failure.
    a2 = "Vacation is unlimited for all employees."
    run_evaluation_pipeline(q2, c2, a2)
