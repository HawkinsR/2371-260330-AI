"""
Demo: Streaming Agents and Pydantic Structured Output
This script demonstrates the production orchestration of Agentic AI: 
1. Utilizing Pydantic for strict, deterministic JSON schemas.
2. Implementing Streaming responses for real-time trainee/user feedback.
3. Managing the Reason + Act (ReAct) loop with token-efficient communication.
"""

import time
import sys
import json
from pydantic import BaseModel, Field
from typing import List, Optional

# =====================================================================
# 1. Structured Output Schema (Pydantic)
# =====================================================================
class FinancialAction(BaseModel):
    """
    Using Pydantic allows us to enforce typing and validation. 
    If the LLM tries to return a string where a float is expected, 
    Pydantic will catch the error before it breaks the application.
    """
    ticker: str = Field(description="The stock ticker symbol (e.g., AAPL)")
    action: str = Field(description="The recommended action: BUY, SELL, or HOLD")
    target_price: float = Field(description="The price at which the action should trigger")
    reasoning: str = Field(description="Short explanation for the recommendation")

# =====================================================================
# 2. Streaming Response Simulation
# =====================================================================
def simulate_streaming_text(message: str, delay: float = 0.03):
    """
    Demonstrates how tokens are sent individually to the client.
    This provides immediate feedback in the UI while the model is still thinking.
    """
    print("   [AGENT STREAMING]: ", end="", flush=True)
    for word in message.split():
        for char in word:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        sys.stdout.write(" ")
        sys.stdout.flush()
        time.sleep(delay)
    print("\n")

# =====================================================================
# 3. The ReAct Agent Loop with Streaming
# =====================================================================
def run_streaming_agent_demo(user_query: str):
    """
    Simulates a production LangChain agent loop with streaming and validation.
    """
    print(f"\n[USER QUERY]: \"{user_query}\"")
    print("-" * 60)

    # --- Step 1: Initial Thought (Streaming) ---
    thought_1 = "I need to analyze Apple stock. First, I'll check the current market volatility and trend."
    simulate_streaming_text(thought_1)

    # --- Step 2: Tool Execution (Simulated) ---
    print("   [ACTION]: Calling 'get_market_data' for ticker 'AAPL'...")
    time.sleep(1)
    print("   [OBSERVATION]: AAPL is trending upward with RSI at 65. Resistance at $235.\n")

    # --- Step 3: Final Reasoning (Streaming) ---
    thought_2 = "Based on the upward trend, I'll recommend a BUY with a target price slightly above current resistance."
    simulate_streaming_text(thought_2)

    # --- Step 4: Structured Output (Pydantic) ---
    print("   [FINAL STEP]: Compiling structured JSON payload via Pydantic...")
    
    # We instantiate the Pydantic model to guarantee the output is valid
    recommendation = FinancialAction(
        ticker="AAPL",
        action="BUY",
        target_price=237.50,
        reasoning="Strong upward momentum and positive RSI suggest a breakout past $235 resistance."
    )

    print("-" * 60)
    print(">>> PRODUCTION PAYLOAD (VALIDATED BY PYDANTIC) <<<\n")
    print(recommendation.model_dump_json(indent=4))
    print("-" * 60)

# =====================================================================
# 4. Execution
# =====================================================================
if __name__ == "__main__":
    print("=== Week 4: Streaming & Structured Agents Demo ===")
    query = "Give me a trade recommendation for Apple based on current technicals."
    run_streaming_agent_demo(query)
