from typing import TypedDict
import json

# =====================================================================
# 1. Simulating the Environment and Tools
# =====================================================================
def tool_calculate_interest(principal: float, rate: float, years: int) -> float:
    print(f"\n   [Tool Execution] calculate_interest(principal={principal}, rate={rate}, years={years})")
    
    interest = principal * rate * years
    total_amount = principal + interest
    
    print(f"   [Tool Result] ${total_amount}")
    return total_amount

# =====================================================================
# 2. Structured Output Schema
# =====================================================================
class LoanReport(TypedDict):
    principal: float
    rate: float
    years: int
    total_amount: float
    summary: str

# =====================================================================
# 3. The ReAct Agent Loop Simulation
# =====================================================================
def simulate_react_agent(user_prompt: str) -> LoanReport:
    print(f"\n[USER PROMPT]: '{user_prompt}'")
    
    system_prompt = """
    You are a strict financial assistant. You MUST use tools to calculate loan math.
    NEVER hallucinate numbers. You MUST return your final answer adhering perfectly 
    to the LoanReport schema.
    """
    print(f"[SYSTEM PROMPT injected securely underneath the user prompt]")
    print("-" * 50)
    
    print("\nAGENT THOUGHT 1: I need to calculate the loan interest. I will use the 'calculate_interest' tool.")
    
    total_amount = tool_calculate_interest(10000.0, 0.05, 3)
    
    print(f"AGENT OBSERVATION 1: The calculated total amount is {total_amount}.")
    print("AGENT THOUGHT 2: I have all the information required. I will format the output.")
    
    final_output: LoanReport = {
        "principal": 10000.0,
        "rate": 0.05,
        "years": 3,
        "total_amount": total_amount,
        "summary": f"Your $10000.0 loan at 5% interest over 3 years yields a total owed of ${total_amount}."
    }
    
    return final_output

# =====================================================================
# 4. Execution
# =====================================================================
if __name__ == "__main__":
    print("=== Agentic AI: Bank Loan Calculator ===")
    
    query = "I am taking out a $10,000 loan at a 5% simple interest rate for 3 years. What is the total I will owe?"
    
    structured_result = simulate_react_agent(query)
    
    print("\n" + "="*50)
    print(">>> FINAL APPLICATION PAYLOAD (GUARANTEED JSON) <<<")
    print(json.dumps(structured_result, indent=4))
    print("="*50 + "\n")
