from unittest.mock import patch
import time

# ==============================================================================
# 1. The Application (The Agent to Test)
# ==============================================================================
def fetch_employee_benefits(query: str) -> str:
    """A very slow tool that hits a live database. DO NOT RUN THIS IN TESTS."""
    print("      [TOOL] Connecting to Live HR Database...")
    time.sleep(5) # Simulating terrible network latency
    
    if "dental" in query.lower():
        return "Employees receive $2,000 in dental coverage annually."
    elif "remote" in query.lower():
        return "Remote Policy: 3 days WFH, 2 days in office."
    return "Database returned no results."

def hr_agent(user_input: str) -> dict:
    """The Agent synthesizes the retrieved data."""
    print(f"\n  [AGENT] Received Input: '{user_input}'")
    
    context = fetch_employee_benefits(user_input)
    
    if "$2,000" in context:
        synthesis = "You are eligible for up to $2,000 in annual dental benefits."
    elif "3 days WFH" in context:
        synthesis = "Employees may work from home 3 days a week."
    else:
        synthesis = "I'm sorry, I couldn't find information regarding your request."
        
    return {"final_answer": synthesis}

# ==============================================================================
# 2. The Golden Dataset and Evaluator
# ==============================================================================

# 1. Complete the Golden Dataset by adding a second test case for remote work.
# Input: "What is the policy for remote work?"
# Expected Output: "Employees may work from home 3 days a week."
GOLDEN_DATASET = [
    {
        "input": "How much dental coverage do I get?",
        "expected_output": "You are eligible for up to $2,000 in annual dental benefits."
    },
    {
        "input": "What is the policy for remote work?",
        "expected_output": "Employees may work from home 3 days a week."
    }
]

def exact_match_evaluator(actual: str, expected: str) -> float:
    return 1.0 if actual == expected else 0.0

# ==============================================================================
# 3. Running the Test Suite WITH Mocking
# ==============================================================================

def run_evaluation_suite():
    print("--- EDD: Running Test Suite ---")
    total_score = 0
    
    # 2. Use python's `patch` context manager to mock '__main__.fetch_employee_benefits'
    with patch('__main__.fetch_employee_benefits') as mock_tool:
        
        # We supply the mocked return values
        mock_tool.side_effect = lambda q: "Employees receive $2,000 in dental coverage annually." if "dental" in q.lower() else "Remote Policy: 3 days WFH, 2 days in office."
        
        for i, scenario in enumerate(GOLDEN_DATASET):
            print(f"\n--- Running Scenario {i+1} ---")
            
            # 3. Invoke the hr_agent with the scenario's input
            result = hr_agent(scenario["input"])
            
            # 4. Extract the actual final_answer and the expected_output
            actual = result["final_answer"]
            expected = scenario["expected_output"]
            
            # 5. Grade the result using exact_match_evaluator and add it to total_score
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
