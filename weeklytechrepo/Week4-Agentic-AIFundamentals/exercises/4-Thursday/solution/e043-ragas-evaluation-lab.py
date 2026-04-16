from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_aws import ChatBedrock, BedrockEmbeddings

# =====================================================================
# 1. Initialize Judge Models
# =====================================================================
judge_llm = ChatBedrock(
    model_id="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    region_name="us-east-1"
)
judge_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1"
)

# =====================================================================
# 2. Build the Evaluation Dataset
# =====================================================================
shared_context = ["Employees receive 15 days of Paid Time Off per calendar year. Carryover is not permitted."]

data_samples = {
    "question": [
        "How many vacation days do employees get?",
        "How many vacation days do employees get?",
    ],
    "answer": [
        # Row A: Faithful — answer is grounded in the context
        "According to company policy, employees receive 15 days of PTO per year and cannot carry days over.",
        # Row B: Hallucinated — answer introduces claims not in context
        "Employees enjoy unlimited vacation days as part of our flexible work culture.",
    ],
    "contexts": [shared_context, shared_context]
}

eval_dataset = Dataset.from_dict(data_samples)

# =====================================================================
# 3. Run the Evaluation
# =====================================================================
def run_exercise():
    print("=== e043: RAGAS Evaluation Lab ===")
    print("Running evaluation... (This makes multiple Bedrock calls, allow ~30s)\n")
    
    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge_llm,
        embeddings=judge_embeddings
    )
    
    df = result.to_pandas()
    
    print("-" * 60)
    print("TEST CASE A — Faithful Answer:")
    print(f"  Faithfulness:     {df.iloc[0]['faithfulness']:.2f}  (Target: >= 0.90)")
    print(f"  Answer Relevancy: {df.iloc[0]['answer_relevancy']:.2f}")

    print("\nTEST CASE B — Hallucinated Answer:")
    print(f"  Faithfulness:     {df.iloc[1]['faithfulness']:.2f}  (Target: <= 0.20)")
    print(f"  Answer Relevancy: {df.iloc[1]['answer_relevancy']:.2f}")
    print("-" * 60)

if __name__ == "__main__":
    run_exercise()
