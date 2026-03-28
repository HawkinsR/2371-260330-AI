"""
Demo: Simple ReAct Agent and Structured Output
This script demonstrates the foundation of Agentic AI: giving an LLM access
to customized Python functions (@tool) and forcing it to solve a problem 
step-by-step using the ReAct (Reason + Act) loop, finally returning structured JSON.
"""

from typing import TypedDict, Annotated
import json

# =====================================================================
# 1. Simulating the Environment and Tools
# =====================================================================
# In LangChain, we would use the @tool decorator. Here we simulate the logic.
def tool_get_stock_price(ticker: str) -> float:
    """Returns the current price of a given stock ticker."""
    print(f"\n   [Tool Execution] get_stock_price(ticker='{ticker}')")
    
    # A simulated database of stock prices
    prices = {"AAPL": 150.0, "TSLA": 200.0, "MSFT": 350.0}
    
    # .get() safely looks up the ticker, returning 0.0 if not found
    price = prices.get(ticker.upper(), 0.0)
    
    print(f"   [Tool Result] ${price}")
    return price

def tool_calculate_portfolio_value(price: float, shares: int) -> float:
    """Multiplies price by shares to get the total portfolio value."""
    print(f"\n   [Tool Execution] calculate_portfolio_value(price={price}, shares={shares})")
    
    # Simple mathematical calculation executed by the tool, not the LLM
    value = price * shares
    
    print(f"   [Tool Result] ${value}")
    return value

# =====================================================================
# 2. Structured Output Schema (Pydantic Simulation)
# =====================================================================
class PortfolioReport(TypedDict):
    """
    We will force the LLM to return EXACTLY this dictionary structure,
    rather than a messy paragraph of text. This is crucial for integrating
    LLMs into traditional software pipelines where expected keys matter.
    """
    ticker_symbol: str
    current_price: float
    shares_owned: int
    total_value: float
    summary: str

# =====================================================================
# 3. The ReAct Agent Loop Simulation
# =====================================================================
def simulate_react_agent(user_prompt: str) -> PortfolioReport:
    """
    Simulates the LangGraph wrapper (create_react_agent) orchestrating 
    the Thought -> Action -> Observation loop.
    """
    print(f"\n[USER PROMPT]: '{user_prompt}'")
    
    # System prompts define the persona, rules, and constraints
    system_prompt = """
    You are a strict financial assistant. You MUST use tools to find stock prices 
    and calculate math. NEVER hallucinate numbers. 
    You MUST return your final answer adhering perfectly to the PortfolioReport schema.
    """
    print(f"[SYSTEM PROMPT injected securely underneath the user prompt]")
    print("-" * 50)
    
    # --- Step 1: Reason ---
    # The LLM reasons about what action to take next based on the prompt
    print("\nAGENT THOUGHT 1: I need to find the price of Apple stock first. I will use the 'get_stock_price' tool.")
    
    # --- Step 2: Act ---
    # The Agent executes the python tool
    price = tool_get_stock_price("AAPL")
    
    # --- Step 3: Observe (Context updated) ---
    # The result of the python execution is added to the AI's short-term memory
    print("AGENT OBSERVATION 1: The price of AAPL is 150.0.")
    print("AGENT THOUGHT 2: Now I need to calculate the total value of 15 shares. I will use the 'calculate_portfolio_value' tool.")
    
    # --- Step 4: Act ---
    # The Agent triggers the second tool, passing in the newly discovered price
    total_value = tool_calculate_portfolio_value(price, 15)
    
    # --- Step 5: Observe (Context updated) ---
    print(f"AGENT OBSERVATION 2: The total value is {total_value}.")
    print("AGENT THOUGHT 3: I have all the information required. I will now format the final output using the required JSON schema.")
    
    # --- Step 6: Final Structured Output ---
    # The LLM returns a structured object rather than raw conversational text
    final_output: PortfolioReport = {
        "ticker_symbol": "AAPL",
        "current_price": price,         # Replaced with tool output
        "shares_owned": 15,             # Extracted from prompt
        "total_value": total_value,     # Replaced with tool output
        "summary": "Your 15 shares of Apple are currently worth $2250.0 based on today's trading price."
    }
    
    return final_output

# =====================================================================
# 4. Execution
# =====================================================================
def run_demo():
    print("=== Agentic AI Fundamentals: ReAct Loop ===")
    
    # The user's query that kicks off the loop
    query = "I own 15 shares of Apple. What is my total portfolio value?"
    
    # Run the agent pipeline
    structured_result = simulate_react_agent(query)
    
    print("\n" + "="*50)
    print(">>> FINAL APPLICATION PAYLOAD (GUARANTEED JSON) <<<")
    
    # Print it elegantly using json.dumps to prove it's a real dictionary/JSON, not a string block
    print(json.dumps(structured_result, indent=4))
    print("="*50 + "\n")

if __name__ == "__main__":
    run_demo()
