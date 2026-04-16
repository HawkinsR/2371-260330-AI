import time
import sys
import json
from typing import Optional
from pydantic import BaseModel, Field

# =====================================================================
# 1. Pydantic Structured Output Schema
# =====================================================================
class LoanCalculation(BaseModel):
    """
    Ensures the LLM's final response adheres to a strict schema.
    """
    principal: float = Field(..., description="The initial loan amount.")
    rate: float = Field(..., description="The simple interest rate (e.g. 0.05 for 5%).")
    years: int = Field(..., description="The duration of the loan in years.")
    total_interest: float = Field(..., description="The calculated interest amount.")
    decision: str = Field(..., description="Final approval status: 'APPROVED' or 'DENIED'.")

# =====================================================================
# 2. Mimicking Token-by-Token Streaming
# =====================================================================
def simulate_streaming_response(text: str):
    """
    Simulates the real-time UX of an LLM.
    """
    print("\n[AI STREAMING]: ", end="", flush=True)
    for word in text.split():
        sys.stdout.write(word + " ")
        sys.stdout.flush()
        time.sleep(0.1) # Simulate token generation delay
    print("\n")

# =====================================================================
# 3. The Orchestration Loop
# =====================================================================
def run_agent_workflow(user_query: str):
    print(f"=== Processing Query: {user_query} ===")
    
    # 1. Simulate Reasoning via Streaming
    reasoning = (
        "Understood. I will calculate the simple interest for this loan request. "
        "I am multiplying the principal by the rate and the time factor. "
        "Checking against bank risk thresholds now..."
    )
    simulate_streaming_response(reasoning)

    # 2. Calculation Logic
    p, r, t = 10000.0, 0.05, 3
    interest = p * r * t
    
    # Risk threshold: Any interest total over $2000 is denied
    status = "DENIED" if interest > 2000 else "APPROVED"

    # 3. Instantiate and Validate via Pydantic
    calculation_result = LoanCalculation(
        principal=p,
        rate=r,
        years=t,
        total_interest=interest,
        decision=status
    )
    
    print("\n>>> FINAL VALIDATED PAYLOAD <<<")
    # Using model_dump_json() ensures the data is clean for a web API
    print(calculation_result.model_dump_json(indent=4))

if __name__ == "__main__":
    # Test Scenario: $10k loan, 5% interest, 3 years
    query = "Calculate a loan for $10,000 at 5% for 3 years."
    run_agent_workflow(query)
