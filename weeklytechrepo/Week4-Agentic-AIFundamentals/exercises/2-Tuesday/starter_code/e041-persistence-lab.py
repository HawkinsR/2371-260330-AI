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
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]

# =====================================================================
# 2. PII Masking Middleware (TODO)
# =====================================================================
def pii_middleware_node(state: AgentState):
    """
    Node that intercepts messages before the model sees them.
    TODO: Search the last HumanMessage for a 16-digit credit card number pattern.
    If found, replace it with '[REDACTED]'.
    Pattern hint: r'\\b(?:\\d[ -]*?){13,16}\\b'
    """
    # YOUR CODE HERE
    return {"messages": []}

# =====================================================================
# 3. Model Node
# =====================================================================
def model_node(state: AgentState):
    # TODO: Initialize ChatBedrock with Claude 3.5 Sonnet
    # Then invoke the model with the current messages
    # Return {"messages": [response]}
    pass

# =====================================================================
# 4. Build the Graph
# =====================================================================
def build_graph():
    # TODO: Create a StateGraph(AgentState)
    # Add two nodes: "middleware" (pii_middleware_node) and "model" (model_node)
    # Set entry_point to "middleware"
    # Add edges: middleware -> model -> END
    
    # TODO: Create a SqliteSaver from sqlite3.connect(":memory:", check_same_thread=False)
    # Compile the graph with the checkpointer
    pass

# =====================================================================
# 5. Execution
# =====================================================================
def run_exercise():
    graph = build_graph()
    config = {"configurable": {"thread_id": "lab-session-001"}}
    
    print("=== e041: Persisting and Securing Agents ===")
    
    # Session 1: Send initial context (with fake PII)
    print("\n--- Session 1 ---")
    s1 = {"messages": [HumanMessage(content="My name is Alex. My card is 4111-2222-3333-4444.")]}
    # TODO: Invoke the graph (not stream) with session 1 input and config

    # Session 2: Ask a follow-up (without re-passing any context)
    print("\n--- Session 2 (Resume) ---")
    s2 = {"messages": [HumanMessage(content="What is my name? And what did I say about my card?")]}
    # TODO: Invoke the graph (not stream) with session 2 input and SAME config
    # Print the final AI message

if __name__ == "__main__":
    run_exercise()
