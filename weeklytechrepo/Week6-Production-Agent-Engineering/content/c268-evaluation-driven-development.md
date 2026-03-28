# Evaluation Driven Development

## Learning Objectives

- Implement rigorous testing philosophies executing Evaluation Driven Development (EDD).
- Cultivate robust baseline metrics by Creating Golden Datasets.
- Execute offline deterministic checks Unit Testing Agents.
- Abstract volatile network layers Mocking Tool Calls cleanly.
- Align human oversight seamlessly tracing Feedback Loops & Annotations.
- Formulate advanced algorithmic oversight establishing Custom Evaluators in LangSmith.

## Why This Matters

When a traditional Software Engineer changes a line of code, they run Unit Tests. If the tests pass, the code goes to production. When an AI Engineer changes a Prompt string, how do they know if the agent just got 10% better or catastrophically worse? If you test manually by sending 5 prompts to the chatbot, you are guessing, not engineering. **Evaluation Driven Development (EDD)** demands that before you change logic in LangGraph, you run the new agent across hundreds of historical scenarios automatically to objectively measure accuracy, latency, and tone shift via Datasets and Evaluator functions.

> **Key Term - Evaluation Driven Development (EDD):** A software development discipline for AI systems where changes to prompts, models, or logic are validated against an automated evaluation suite *before* deployment. EDD is the AI equivalent of Test-Driven Development (TDD) — you define measurable quality criteria first, then iterate until your agent meets them consistently.

## The Concept

### Golden Datasets

A "Golden Dataset" in LangSmith is a collection of Key-Value pairs representing inputs and expected outputs.
`Input`: "How many vacation days do I get?"
`Expected Output`: "15 days. [Source: handbook.pdf]"
Over time, as edge cases arise in production, you manually augment this dataset. It acts as the ultimate regression test suite.

> **Key Term - Golden Dataset:** A curated, human-verified collection of (input, expected output) pairs used as the ground truth for evaluating an AI system. Every time you modify a prompt or swap a model, you run your agent against the Golden Dataset to check for regressions. New edge cases discovered in production are added to the dataset over time, making it progressively stronger.

> **Key Term - Regression Testing:** Running an existing test suite after making changes to verify that previously working behavior has not broken. In AI development, a regression occurs when a prompt change that improves one behavior accidentally degrades another — for example, making the bot more concise but causing it to drop citations.

### Custom Evaluators and LLM-as-a-Judge

It is easy to assert that `2+2 == 4`. It is difficult to assert that an AI wrote a "polite email."
To evaluate non-deterministic AI outputs against Golden Datasets, we use **LLM-as-a-Judge**. We write a separate Python function (an Evaluator) that calls an LLM (like GPT-4). We pass the prompt, the agent's actual response, and the expected golden answer to the Evaluator LLM, asking it: "Score the agent's response from 1-5 on accuracy and politeness."
LangSmith orchestrates this automatically logging all evaluated scores to a central dashboard.

> **Key Term - LLM-as-a-Judge:** A technique where a separate, powerful LLM (the "judge") is used to evaluate the quality of another AI system's output. Because AI outputs are often subjective (tone, helpfulness, creativity), deterministic string-matching tests fail. The judge LLM reads the question, the expected answer, and the actual agent output, then scores and explains the result — automating what would otherwise require human reviewers.

### Mocking Tool Calls for Offline Testing

Running a test suite of 500 scenarios that actually hits live databases and external APIs is slow, expensive, and dangerous. Standard Python `unittest.mock` allows us to "patch" the agent's tools during evaluation. When the agent attempts to search Pinecone, the mock simply returns a static, hardcoded dictionary immediately. This isolates the test to measure exclusively whether the LLM reasoned correctly, without involving network latency.

> **Key Term - Mocking / Patching:** A testing technique where a real dependency (database, API call, external service) is replaced with a fake implementation that returns pre-defined, controlled data. Mocking isolates the unit under test — in AI evaluation, this means measuring whether the *LLM reasoned correctly* independently of whether the *network and database worked correctly*.

## Code Example

```python
from langsmith import Client
from langsmith.evaluation import evaluate

# Initialize LangSmith Client (Requires LANGCHAIN_API_KEY env var)
client = Client()

# 1. Define the Application (Your LangGraph or chain)
def sample_agent(inputs: dict) -> dict:
    """Mock agent representing an LLM chain."""
    user_query = inputs["question"]
    # In reality, this would invoke your LLM
    if "vacation" in user_query.lower():
        return {"answer": "You receive 15 days of PTO annually."}
    return {"answer": "I do not know."}

# 2. Define a Custom Evaluator Function
def exact_match_evaluator(run, example) -> dict:
    """
    Evaluates if the agent's output exactly matches the expected Golden Output.
    (In production, you would typically use an LLM-as-a-Judge here instead for nuance).
    """
    actual_output = run.outputs["answer"]
    expected_output = example.outputs["correct_answer"]
    
    score = 1.0 if actual_output == expected_output else 0.0
    return {"key": "exact_match_score", "score": score}

# 3. Create a Golden Dataset programmatically (or via the UI)
# Note: You only run dataset creation once to seed the platform!
"""
dataset = client.create_dataset(dataset_name="Vacation_Policy_Tests")
client.create_example(
    inputs={"question": "How many vacation days do I get?"},
    outputs={"correct_answer": "You receive 15 days of PTO annually."},
    dataset_id=dataset.id
)
"""

# 4. Execute the Evaluation Suite
# Warning: This tries to connect to the LangSmith cloud backend.
try:
    results = evaluate(
        sample_agent,  # The application being tested
        data="Vacation_Policy_Tests", # The name of the Golden Dataset in LangSmith
        evaluators=[exact_match_evaluator], # Array of functions grading the output
        experiment_prefix="test-sprint-v2"
    )
    print("Evaluation triggered successfully. Check LangSmith Dashboard!")
except Exception as e:
    print(f"Skipping live evaluation due to config: {e}")

# --- Mocking Pattern (Offline / CI Testing) ---
# This pattern lets you test your agent's logic WITHOUT connecting to live APIs:
from unittest.mock import patch

def search_pinecone(query: str) -> list:
    """The real tool — would hit a live Pinecone index."""
    raise ConnectionError("Not available in test environment")

def agent_that_uses_search(query: str) -> str:
    docs = search_pinecone(query)  # Agent calls the tool
    return docs[0]["content"] if docs else "No results"

# In your test, `patch` replaces `search_pinecone` with a fake:
with patch("__main__.search_pinecone") as mock_search:
    mock_search.return_value = [{"content": "You receive 15 days of PTO annually."}]
    result = agent_that_uses_search("How many vacation days?")
    assert result == "You receive 15 days of PTO annually."
    print(f"Mock test passed. Agent returned: '{result}'")
    # The real Pinecone was never called — test is fast, free, and deterministic.
```

## Additional Resources

- [LangSmith Evaluation Concepts](https://docs.smith.langchain.com/evaluation)
- [LLM-as-a-Judge Approaches](https://docs.smith.langchain.com/evaluation/evaluators)
