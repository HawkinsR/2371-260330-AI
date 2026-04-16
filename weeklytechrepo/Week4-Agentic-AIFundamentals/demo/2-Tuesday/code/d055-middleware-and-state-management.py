import sqlite3
import re
from typing import Annotated, TypedDict
from langchain_aws import ChatBedrock
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# =====================================================================
# 1. State Definition
# =====================================================================
class AgentState(TypedDict):
    # The state contains a list of messages
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]

# =====================================================================
# 2. Middleware Logic (PII Masking)
# =====================================================================
def pii_masking_middleware(state: AgentState):
    """
    middleware node that scans the last human message for PII (Credit Cards)
    and redacts them before the LLM sees the history.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        # Simple regex for a 16-digit card simulation
        pattern = r"\b(?:\d[ -]*?){13,16}\b"
        content = last_message.content
        if re.search(pattern, content):
            print("   [SECURITY]: PII detected! Redacting sensitive information...")
            new_content = re.sub(pattern, "[REDACTED_CARD]", content)
            # We override the message in the transition
            last_message.content = new_content
    
    return {"messages": []} # Return empty since we modified the message in place

# =====================================================================
# 3. Initialize Model and Graph
# =====================================================================
def get_persistent_agent():
    # 1. Setup DB Persistence
    conn = sqlite3.connect("agent_persistence.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # 2. Setup Model
    model = ChatBedrock(
        model_id="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        region_name="us-east-1"
    )

    # 3. Define the Graph
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("middleware", pii_masking_middleware)
    workflow.add_node("model", lambda state: {"messages": [model.invoke(state["messages"])]})

    # Define Edges
    workflow.set_entry_point("middleware")
    workflow.add_edge("middleware", "model")
    workflow.add_edge("model", END)

    # Compile with persistence
    return workflow.compile(checkpointer=memory if 'memory' in locals() else checkpointer)

# =====================================================================
# 4. Execution
# =====================================================================
def run_demo():
    agent = get_persistent_agent()
    
    # We use a consistent thread_id to demonstrate 'memory'
    config = {"configurable": {"thread_id": "session_alpha_99"}}

    print("--- [SESSION 1: Initial Query with PII] ---")
    query_1 = {"messages": [HumanMessage(content="My name is Richard and my card number is 4111-2222-3333-4444. What is my name?")]}
    
    for chunk in agent.stream(query_1, config, stream_mode="values"):
        msg = chunk["messages"][-1]
        print(f"[{msg.type.upper()}]: {msg.content}")

    print("\n--- [SESSION 2: Resumption (No history passed)] ---")
    # Notice we don't pass 'Richard' or the card here, the agent loads it from SQLite
    query_2 = {"messages": [HumanMessage(content="What was the card number I gave you earlier?")]}
    
    for chunk in agent.stream(query_2, config, stream_mode="values"):
        msg = chunk["messages"][-1]
        print(f"[{msg.type.upper()}]: {msg.content}")

if __name__ == "__main__":
    print("=== Week 4: Middleware & Persistence Demo ===")
    run_demo()
