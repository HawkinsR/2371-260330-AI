from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# =====================================================================
# 1. Initialize Judge Models
# =====================================================================
# TODO: Import ChatBedrock from langchain_aws and BedrockEmbeddings.
# Initialize:
#   - judge_llm using Claude 3.5 Sonnet (model_id="us.anthropic.claude-3-5-sonnet-20240620-v1:0")
#   - judge_embeddings using Titan v2 (model_id="amazon.titan-embed-text-v2:0")

# from langchain_aws import ChatBedrock, BedrockEmbeddings
# judge_llm = ...
# judge_embeddings = ...

# =====================================================================
# 2. Build the Evaluation Dataset
# =====================================================================
# The dataset must have these columns:
#   - "question": List[str] — The user's question
#   - "answer": List[str] — The LLM's answer
#   - "contexts": List[List[str]] — The retrieved context documents used

shared_context = ["Employees receive 15 days of Paid Time Off per calendar year. Carryover is not permitted."]

data_samples = {
    "question": [
        "How many vacation days do employees get?",  
        "How many vacation days do employees get?",
    ],
    "answer": [
        # Row A: TODO — Write a faithful answer that references ONLY what's in shared_context
        "TODO: Write a faithful answer here.",
        # Row B: TODO — Write a hallucinated answer (e.g., claim unlimited days)
        "TODO: Write a hallucinated answer here.",
    ],
    "contexts": [
        shared_context,
        shared_context,
    ]
}

# TODO: Create a HuggingFace Dataset from the dict above
# eval_dataset = Dataset.from_dict(data_samples)

# =====================================================================
# 3. Run the Evaluation
# =====================================================================
def run_exercise():
    print("=== e043: RAGAS Evaluation Lab ===\n")
    
    # TODO: Call ragas.evaluate() with:
    #   - dataset=eval_dataset
    #   - metrics=[faithfulness, answer_relevancy]
    #   - llm=judge_llm
    #   - embeddings=judge_embeddings
    print("Note: Each evaluation call makes multiple LLM calls. This may take 15-30 seconds.")
    
    # result = evaluate(...)
    
    # TODO: Convert to a Pandas DataFrame and print Faithfulness/Relevancy for each row.
    # df = result.to_pandas()
    # print(df[["question", "faithfulness", "answer_relevancy"]])

if __name__ == "__main__":
    run_exercise()
