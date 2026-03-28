from typing import TypedDict
import json

# =====================================================================
# 1. Simulating the Environment and Tools
# =====================================================================
def tool_calculate_interest(principal: float, rate: float, years: int) -> float:
    """Calculates simple interest and returns the total amount owed."""
    print(f"\n   [Tool Execution] calculate_interest(principal={principal}, rate={rate}, years={years})")
    
    # 1. TODO: Calculate interest = principal * rate * years
    interest = 0.0
    
    # 2. TODO: Calculate total_amount = principal + interest
    total_amount = 0.0
    
    print(f"   [Tool Result] ${total_amount}")
    return total_amount

# =====================================================================
# 2. Structured Output Schema
# =====================================================================
class LoanReport(TypedDict):
    """
    We will force the LLM to return EXACTLY this dictionary structure.
    """
    # 3. TODO: Add the typed fields: 
    # principal (float), rate (float), years (int), total_amount (float), summary (str)
    
    pass

# =====================================================================
# 3. The ReAct Agent Loop Simulation
# =====================================================================
def simulate_react_agent(user_prompt: str) -> LoanReport:
    """
    Simulates the LangGraph wrapper orchestrating the ReAct loop.
    """
    print(f"\n[USER PROMPT]: '{user_prompt}'")
    
    # 4. TODO: Define a strict system prompt
    system_prompt = """
    
    """
    print(f"[SYSTEM PROMPT injected securely underneath the user prompt]")
    print("-" * 50)
    
    # --- Step 1: Reason ---
    print("\nAGENT THOUGHT 1: I need to calculate the loan interest. I will use the 'calculate_interest' tool.")
    
    # --- Step 2: Act ---
    # 5. TODO: Call the tool_calculate_interest function with (10000.0, 0.05, 3)
    total_amount = None
    
    # --- Step 3: Observe (Context updated) ---
    print(f"AGENT OBSERVATION 1: The calculated total amount is {total_amount}.")
    print("AGENT THOUGHT 2: I have all the information required. I will format the output.")
    
    # --- Step 4: Final Structured Output ---
    # 6. TODO: Populate the final output matching the LoanReport schema
    final_output: LoanReport = {
        
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
    
    # If the dictionary is still empty, this will just print {}
    try:
        print(json.dumps(structured_result, indent=4))
    except TypeError:
        print(structured_result)
        
    print("="*50 + "\n")
