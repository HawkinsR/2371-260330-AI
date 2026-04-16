import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_aws import ChatBedrock, BedrockEmbeddings

# =====================================================================
# 1. Pipeline Initialization (The LLM-as-a-Judge)
# =====================================================================
def get_evaluator_models():
    """Initialize the LLM and Embeddings that Ragas will use to score the pipelines."""
    # The Judge LLM (Using Claude 3.5 Sonnet for high reasoning capabilities)
    judge_llm = ChatBedrock(
        model_id="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        region_name="us-east-1"
    )
    
    # The Embedding model used by Ragas to calculate distance and relevancy
    judge_embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-east-1"
    )
    
    return judge_llm, judge_embeddings

# =====================================================================
# 2. RAGAS Evaluation Execution
# =====================================================================
def run_demo():
    print("=== Week 4: RAGAS Evaluation Pipeline (Amazon Bedrock) ===")
    
    # Ensure AWS Credentials are in the environment before proceeding
    if "AWS_REGION" not in os.environ:
        os.environ["AWS_REGION"] = "us-east-1"
        
    try:
        judge_llm, judge_embeddings = get_evaluator_models()
    except Exception as e:
        print(f"\n[ERROR]: Failed to connect to AWS Bedrock. Details: {e}")
        return

    print("\n[RAGAS]: Initializing LLM-as-a-Judge Evaluation Engine...")

    # --- Step 1: Create the Evaluation Dataset ---
    # In production, this data comes from your LangSmith logs or production RAG pipeline.
    # The dataset requires: question, answer, and contexts (List[str]).
    
    data_samples = {
        "question": [
            "What is the vacation policy?",
            "What is the vacation policy?"
        ],
        "answer": [
            # Case A: Accurate and faithful
            "According to the handbook, you receive 15 days of PTO.",
            # Case B: Hallucinated (Claiming unlimited PTO)
            "You have unlimited vacation days because we value work-life balance."
        ],
        "contexts": [
            ["Employees receive 15 days of PTO per calendar year. Carrying over is not permitted."],
            ["Employees receive 15 days of PTO per calendar year. Carrying over is not permitted."]
        ]
    }

    eval_dataset = Dataset.from_dict(data_samples)

    # --- Step 2: Run the Evaluation ---
    print("\n--- TEST EXECUTION: Analyzing Faithfulness and Relevancy ---")
    print("This requires multiple LLM calls per row, so this may take ~10-20 seconds...\n")
    
    try:
        # We pass our Bedrock models explicitly to Ragas
        result = evaluate(
            dataset=eval_dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=judge_llm,
            embeddings=judge_embeddings
        )
        
        # --- Step 3: Analyze Results ---
        df = result.to_pandas()
        
        print("-" * 60)
        print("TEST CASE A (Accurate Answer):")
        print(f"  - Faithfulness: {df.iloc[0]['faithfulness']:.2f} (Expected: 1.0)")
        print(f"  - Relevancy: {df.iloc[0]['answer_relevancy']:.2f}")
        
        print("\nTEST CASE B (Hallucination):")
        print(f"  - Faithfulness: {df.iloc[1]['faithfulness']:.2f} (Expected: 0.0)")
        print(f"  - Relevancy: {df.iloc[1]['answer_relevancy']:.2f}")
        print("-" * 60)
        
    except Exception as e:
        print(f"[ERROR during Ragas Evaluation]: {e}")

if __name__ == "__main__":
    run_demo()
