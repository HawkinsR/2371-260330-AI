import sys
from pydantic import BaseModel, Field
from langchain_aws import ChatBedrock
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

# =====================================================================
# 1. Structured Output Schema (Pydantic)
# =====================================================================
class StockAnalysis(BaseModel):
    """Schema for financial analysis results."""
    ticker: str = Field(description="The stock ticker symbol")
    recommendation: str = Field(description="BUY, SELL, or HOLD")
    confidence_score: float = Field(description="Confidence from 0.0 to 1.0")
    reasoning: str = Field(description="Technical justification for the action")

# =====================================================================
# 2. Tool Definition (Real-style)
# =====================================================================
@tool
def get_market_sentiment(ticker: str) -> str:
    """Retrieves current market sentiment scores for a given stock ticker."""
    # Simulation of a real API call (e.g. to AlphaVantage or NewsAPI)
    data = {
        "AAPL": "Sentiment is highly positive (+0.8) following iPhone sales reports.",
        "TSLA": "Sentiment is mixed (-0.2) due to regulatory concerns.",
        "AMZN": "Sentiment is neutral (+0.1) awaiting earnings."
    }
    return data.get(ticker.upper(), "Sentiment data unavailable.")

# =====================================================================
# 3. Initialize Amazon Bedrock Model
# =====================================================================
def get_bedrock_agent():
    # Use the universal initializer to target Amazon Bedrock
    # Claude 3.5 Sonnet is ideal for reasoning and tool-use
    llm = init_chat_model(
        model="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        model_provider="bedrock",
        temperature=0 # Deterministic responses for tool calls
    )

    tools = [get_market_sentiment]
    system_prompt = "You are a professional financial analyst. Always use tools to verify sentiment before recommending an action."
    
    # Create the ReAct agent
    return create_react_agent(llm, tools, state_modifier=system_prompt)

# =====================================================================
# 4. Execution with Streaming
# =====================================================================
def run_demo():
    agent = get_bedrock_agent()
    query = {"messages": [("user", "What is your recommendation for Apple (AAPL) stock?")]}
    
    print("--- [STARTING AGENTIC LOOP] ---")
    
    # Use stream_mode="values" to see how state evolves
    for chunk in agent.stream(query, stream_mode="values"):
        last_message = chunk["messages"][-1]
        
        # We handle different message types for clear display
        if last_message.type == "human":
            print(f"\n[USER]: {last_message.content}")
        elif last_message.type == "ai" and last_message.tool_calls:
            print(f"\n[AI THOUGHT]: {last_message.content or 'Thinking...'}")
            for tool_call in last_message.tool_calls:
                print(f"   [TOOL CALL]: {tool_call['name']}({tool_call['args']})")
        elif last_message.type == "tool":
            print(f"   [TOOL RESULT]: {last_message.content}")
        elif last_message.type == "ai":
            print(f"\n[FINAL RESPONSE]: {last_message.content}")

    print("\n--- [DEMO COMPLETE] ---")

if __name__ == "__main__":
    # Ensure AWS Credentials are set in environment
    # import os
    # os.environ["AWS_ACCESS_KEY_ID"] = "..."
    # os.environ["AWS_SECRET_ACCESS_KEY"] = "..."
    # os.environ["AWS_REGION"] = "us-east-1"
    
    run_demo()
