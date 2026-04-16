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
        
        # 1. Faithfulness Scoring Logic
        # Does the answer stay grounded in the provided context?
        faithfulness = 1.0
        # Scenario: If answer contains 'unlimited' but context says '15 days'
        if "unlimited" in answer.lower() and "15 days" in context.lower():
            faithfulness = 0.0
        
        # 2. Answer Relevancy Scoring Logic
        # Does the answer actually address the user's question?
        relevancy = 0.0
        if "vacation" in query.lower() and "vacation" in answer.lower():
            relevancy = 0.95
        
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
        # 3. Log a 'CRITICAL' alert if faithfulness score is below 0.5.
        if metric == "faithfulness" and score < 0.5:
            print(f"   [!!! CRITICAL ALERT !!!]: Hallucination Detected (Faithfulness: {score})")
            status = "FAIL"
        else:
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
    # 4. Scenario: The model hallucinates 'unlimited' vacation
    a2 = "According to company policy, vacation is unlimited for all employees."
    run_evaluation_pipeline(q2, c2, a2)
