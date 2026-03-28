"""
Demo: Mocking Tools and Unit Testing Agents
This script demonstrates how to evaluate an agent offline against a "Golden Dataset".
It shows how to mock a slow, expensive external API call to ensure the test suite
runs instantly and deterministically.
"""

from unittest.mock import patch
import json

# ==============================================================================
# 1. The Application (The Agent we want to test)
# ==============================================================================
def live_database_search_tool(query: str) -> str:
    """
    Imagine this connects to a live SQL database or Pinecone Vector DB.
    It takes 3 seconds to run and costs money. We DO NOT want this 
    running during our offline CI/CD test suite.
    """
    import time
    print(f"      [TOOL] Connecting to Live Database for query: '{query}'...")
    time.sleep(1) # Simulating network latency
    
    # In a real app, this would execute a SQL query or Vector search
    if "vacation" in query.lower():
        return "Company policy grants 15 days of PTO annually."
    return "Database returned no results."

# A BUG WAS FOUND HERE PREVIOUSLY, WE NEED TO MAKE SURE THE AGENT TAKES AN INPUT
# AND SYNTHESIZES A RESPONSE BASED ON THE TOOL OUTPUT
def customer_service_agent(user_input: str) -> dict:
    """
    A simple agent that takes user input, uses a tool to find facts,
    and returns a formatted JSON response.
    """
    print(f"\n  [AGENT] Received Input: '{user_input}'")
    
    # The Agent decides to use the tool
    print(f"  [AGENT] I need more context. Invoking search tool...")
    context = live_database_search_tool(user_input)
    
    # The Agent synthesizes the final answer based on the context returned
    print(f"  [AGENT] Synthesizing final response...")
    if "15 days" in context:
        synthesis = "Based on our records, you are entitled to 15 days of paid time off."
    else:
        synthesis = "I'm sorry, I couldn't find information regarding your request."
        
    return {"final_answer": synthesis}

# ==============================================================================
# 2. Evaluation Framework (Simulating LangSmith EDD)
# ==============================================================================

# A "Golden Dataset" - Hardcoded inputs and the exact outputs we expect from a perfect agent.
# This serves as our ground truth for testing.
GOLDEN_DATASET = [
    {
        "input": "How many vacation days do I get this year?",
        "expected_output": "Based on our records, you are entitled to 15 days of paid time off."
    },
    {
        "input": "What is the capital of France?",
        "expected_output": "I'm sorry, I couldn't find information regarding your request."
    }
]

def exact_match_evaluator(actual_output: str, expected_output: str) -> float:
    """
    A strictly deterministic evaluator. 1.0 = Perfect. 0.0 = Complete failure.
    (In a real scenario, you'd use LLM-as-a-Judge to grade semantic similarity).
    """
    if actual_output == expected_output:
        return 1.0 # The agent's output exactly matches the golden dataset
    return 0.0     # The agent hallucinated or gave the wrong format

# ==============================================================================
# 3. Running the Test Suite WITH Mocking
# ==============================================================================

def run_evaluation_suite():
    print("--- Evaluation Driven Development (EDD) Demo ---")
    print("Initiating test suite against Golden Dataset...")
    
    total_score = 0
    
    # We use 'patch' to temporarily replace 'live_database_search_tool' 
    # with a fake function (mock) that runs instantly and always returns what we want.
    # The '__main__' part tells Python to patch it in the current running script.
    with patch('__main__.live_database_search_tool') as mock_tool:
        
        # Configure the mock to return exactly what the LLM expects from the database.
        # This isolates the test to ONLY measure if the Agent synthesized the answer correctly,
        # without worrying if the database is down or slow.
        mock_tool.side_effect = lambda q: "Company policy grants 15 days of PTO annually." if "vacation" in q.lower() else "No results."
        
        for i, scenario in enumerate(GOLDEN_DATASET):
            print(f"\n--- Running Scenario {i+1} ---")
            
            # Run the agent (which will hit the mock tool, NOT the real database)
            result = customer_service_agent(scenario["input"])
            
            # Evaluate the output against the Golden Dataset
            actual = result["final_answer"]
            expected = scenario["expected_output"]
            
            # Grade the agent's performance
            score = exact_match_evaluator(actual, expected)
            total_score += score
            
            print(f"  [EVALUATOR] Actual Output:   '{actual}'")
            print(f"  [EVALUATOR] Expected Output: '{expected}'")
            print(f"  [EVALUATOR] Scenario Score:  {score * 100}%")
            
    # Final Aggregate Metrics
    aggregate = (total_score / len(GOLDEN_DATASET)) * 100
    print("\n" + "="*50)
    print(f">>> TEST SUITE COMPLETED. AGGREGATE SCORE: {aggregate}% <<<")
    if aggregate == 100:
        print(">>> RESULT: PASS. Agent is verified for Production deployment.")
    else:
        print(">>> RESULT: FAIL. Regression detected. Do not deploy.")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_evaluation_suite()
