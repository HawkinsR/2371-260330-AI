from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

# =====================================================================
# 1. Pydantic Structured Output Schema
# =====================================================================
class StockRecommendation(BaseModel):
    ticker: str = Field(description="The stock ticker symbol, e.g. TSLA")
    recommendation: str = Field(description="Investment action: BUY, SELL, or HOLD")
    reasoning: str = Field(description="One-sentence technical justification")

# =====================================================================
# 2. Tool Definition
# =====================================================================
@tool
def get_stock_sentiment(ticker: str) -> str:
    """
    Retrieves current market sentiment for a given stock ticker.
    Use this tool whenever the user asks for a stock recommendation or price outlook.
    """
    sentiment_data = {
        "AAPL": "Strong positive sentiment (+0.85). iPhone Super Cycle expectations driving bullish momentum.",
        "TSLA": "Mixed sentiment (-0.15). Strong EV delivery numbers offset by macro rate concerns.",
        "AMZN": "Neutral sentiment (+0.30). AWS growth solid; retail margin pressure remains.",
    }
    return sentiment_data.get(
        ticker.upper(),
        f"No sentiment data found for {ticker}. Use general market context."
    )

# =====================================================================
# 3. Agent Initialization
# =====================================================================
llm = init_chat_model(
    model="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_provider="bedrock",
    temperature=0
)

# =====================================================================
# 4. Create the ReAct Agent
# =====================================================================
system_prompt = (
    "You are a professional financial analyst. "
    "Always use the get_stock_sentiment tool before making any recommendation. "
    "Provide clear, evidence-based BUY, SELL, or HOLD recommendations."
)
agent = create_react_agent(llm, tools=[get_stock_sentiment], state_modifier=system_prompt)

# =====================================================================
# 5. Stream the Agent Response
# =====================================================================
def run_exercise():
    print("=== e040: Your First Bedrock Agent ===\n")
    query = {"messages": [("user", "What is your recommendation for Tesla (TSLA) stock?")]}
    
    for chunk in agent.stream(query, stream_mode="values"):
        last = chunk["messages"][-1]
        if last.type == "ai" and last.tool_calls:
            print(f"[AI THOUGHT]: Calling tool -> {last.tool_calls[0]['name']}({last.tool_calls[0]['args']})")
        elif last.type == "tool":
            print(f"[TOOL RESULT]: {last.content}")
        elif last.type == "ai":
            print(f"\n[FINAL RECOMMENDATION]: {last.content}")

if __name__ == "__main__":
    run_exercise()
