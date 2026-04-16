import time
import sys
import json
from typing import Optional
from pydantic import BaseModel, Field

# =====================================================================
# 1. Pydantic Structured Output Schema
# =====================================================================
# TODO: Define a 'LoanCalculation' class inheriting from Pydantic's BaseModel.
# Fields: principal (float), rate (float), years (int), total_interest (float), decision (str)
class LoanCalculation(BaseModel):
    """
    Ensures the LLM's final response adheres to a strict schema.
    """
    pass

# =====================================================================
# 2. Mimicking Token-by-Token Streaming
# =====================================================================
def simulate_streaming_response(text: str):
    """
    Simulates the real-time UX of an LLM.
    """
    print("\n[AI STREAMING]: ", end="", flush=True)
    # TODO: Loop through each word in 'text'.
    # Use sys.stdout.write(word + " ") and sys.stdout.flush().
    # Use time.sleep(0.1) to simulate network latency.
    print("\n")

# =====================================================================
# 3. The Orchestration Loop
# =====================================================================
def run_agent_workflow(user_query: str):
    print(f"=== Processing Query: {user_query} ===")
    
    # 1. Simulate Reasoning via Streaming
    reasoning = "I will calculate the interest for a $10,000 loan at 5% over 3 years. The formula is P * R * T."
    simulate_streaming_response(reasoning)

    # 2. TODO: Perform the calculation logic (10000 * 0.05 * 3)
    p = 10000.0
    r = 0.05
    t = 3
    interest = 0.0 # Calculate here

    # 3. TODO: Instantiate the Pydantic model with the results.
    # If interest > 2000, decision is 'DENIED', otherwise 'APPROVED'.
    
    # calculation_result = LoanCalculation(...)
    
    # print("\n>>> FINAL VALIDATED PAYLOAD <<<")
    # print(calculation_result.model_dump_json(indent=4))

if __name__ == "__main__":
    query = "Calculate a loan for $10,000 at 5% for 3 years."
    run_agent_workflow(query)
