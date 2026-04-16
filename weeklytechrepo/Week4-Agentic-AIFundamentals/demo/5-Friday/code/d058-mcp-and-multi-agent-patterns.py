import sqlite3
from typing import Annotated, TypedDict, Literal
from langchain_aws import ChatBedrock
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

# =====================================================================
# 1. Shared State Definition
# =====================================================================
class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    # Which specialized agent to route to next
    next_agent: str

# =====================================================================
# 2. Specialized Tools
# =====================================================================
@tool
def search_company_news(company: str) -> str:
    """Searches for the latest news and reports for a given company."""
    return f"[WEB SEARCH RESULT]: {company} reported strong Q1 earnings, beating analyst expectations by 12%."

@tool
def get_stock_price(ticker: str) -> str:
    """Retrieves the current real-time stock price for a given ticker."""
    data = {"AAPL": "$232.15", "AMZN": "$187.40", "TSLA": "$175.90"}
    return f"Current price of {ticker.upper()}: {data.get(ticker.upper(), 'Not found.')}"

# =====================================================================
# 3. Worker Agents (Specialized LLMs)
# =====================================================================
def create_worker_agents():
    llm = ChatBedrock(
        model_id="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        region_name="us-east-1"
    )
    
    # Research Agent — focused purely on news and information
    research_agent = create_react_agent(
        llm, 
        [search_company_news],
        state_modifier="You are a Research Specialist. Only use the search tool. Be concise."
    )
    
    # Finance Agent — focused purely on financial data
    finance_agent = create_react_agent(
        llm, 
        [get_stock_price],
        state_modifier="You are a Finance Specialist. Only retrieve stock prices. Be concise."
    )
    
    return research_agent, finance_agent

# =====================================================================
# 4. Supervisor Graph
# =====================================================================
def build_supervisor_graph():
    research_agent, finance_agent = create_worker_agents()
    
    def router_node(state: SupervisorState):
        """Supervisor logic: decides which worker handles the next step."""
        last_message = state["messages"][-1].content.lower()
        if "news" in last_message or "report" in last_message:
            return {"next_agent": "research"}
        elif "price" in last_message or "stock" in last_message:
            return {"next_agent": "finance"}
        return {"next_agent": "finish"}

    def research_node(state: SupervisorState):
        result = research_agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"][-1:]}

    def finance_node(state: SupervisorState):
        result = finance_agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"][-1:]}

    def verify_action_node(state: SupervisorState):
        """HITL Node: execution pauses here for human review."""
        print("\n   [HITL CHECKPOINT]: Action requires human approval. Execution paused.")
        print("   [SYSTEM]: State saved to DB. Resume via: agent.invoke(None, config)")
        return {}

    def route_decision(state: SupervisorState) -> Literal["research", "finance", "verify_action", END]:
        return state.get("next_agent", "finish") if state.get("next_agent") != "finish" else END

    # Build the Graph
    workflow = StateGraph(SupervisorState)
    workflow.add_node("router", router_node)
    workflow.add_node("research", research_node)
    workflow.add_node("finance", finance_node)
    workflow.add_node("verify_action", verify_action_node)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges("router", route_decision)
    workflow.add_edge("research", "verify_action")
    workflow.add_edge("finance", "verify_action")
    workflow.add_edge("verify_action", END)

    # Compile with persistence AND interrupt
    conn = sqlite3.connect("supervisor_state.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["verify_action"]  # HITL: pause before taking action
    )

# =====================================================================
# 5. Execution
# =====================================================================
def run_demo():
    print("=== Week 4: Multi-Agent Orchestration & HITL Demo ===")
    graph = build_supervisor_graph()
    config = {"configurable": {"thread_id": "supervisor-run-001"}}

    # --- Session 1: Start (Agent will pause at HITL node) ---
    print("\n--- SESSION 1: Initial Request (Will be paused for Human Approval) ---")
    query = {"messages": [HumanMessage(content="Get me the latest news on Apple.")]}
    
    for chunk in graph.stream(query, config, stream_mode="values"):
        last = chunk["messages"][-1]
        print(f"[{last.type.upper()}]: {last.content}")

    print("\n[SYSTEM]: Agent is paused. A human must approve continuation.")
    print("[SYSTEM]:", graph.get_state(config).next) 

    # --- Session 2: Resume (Human clicked 'Approve') ---
    print("\n--- SESSION 2: Human Approved — Resuming Agent ---")
    # Passing None as input resumes from the saved checkpoint
    for chunk in graph.stream(None, config, stream_mode="values"):
        last = chunk["messages"][-1]
        print(f"[{last.type.upper()}]: {last.content}")
    
    print("\n=== DEMO COMPLETE ===")

if __name__ == "__main__":
    run_demo()
