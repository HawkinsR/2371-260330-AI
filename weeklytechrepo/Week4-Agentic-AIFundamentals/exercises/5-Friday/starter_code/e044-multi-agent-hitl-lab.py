import sqlite3
from typing import Annotated, TypedDict, Literal
from langchain_aws import ChatBedrock
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

# =====================================================================
# 1. Shared State
# =====================================================================
class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    next_agent: str

# =====================================================================
# 2. Worker Tools (TODO)
# =====================================================================
# TODO: Define a @tool function 'search_policy(topic: str) -> str'
# It should accept a topic like "remote work" or "PTO" and return a mock policy string.

# TODO: Define a @tool function 'get_employee_data(emp_id: str) -> str'
# It should return mock employee info for a given ID.

# =====================================================================
# 3. Worker Agents (TODO)
# =====================================================================
# TODO: Initialize a ChatBedrock model (Claude 3.5 Sonnet)
# TODO: Create a 'policy_agent' using create_react_agent with only [search_policy]
# TODO: Create an 'hr_agent' using create_react_agent with only [get_employee_data]

# =====================================================================
# 4. Supervisor Graph Nodes (TODO)
# =====================================================================
def router_node(state: SupervisorState):
    """Routes to the correct worker based on message keywords."""
    # TODO: Check the last message content.
    # If "policy" is in the message, set next_agent="policy"
    # If "employee" or "emp" is in the message, set next_agent="hr"
    # Otherwise, set next_agent="finish"
    return {"next_agent": "finish"}

def policy_node(state: SupervisorState):
    # TODO: Invoke the policy_agent with current messages, return the last message
    pass

def hr_node(state: SupervisorState):
    # TODO: Invoke the hr_agent with current messages, return the last message
    pass

def verify_node(state: SupervisorState):
    """HITL checkpoint — graph pauses here for human review."""
    print("\n   [HITL]: Action pending approval. State saved.")
    return {}

def route_decision(state: SupervisorState) -> Literal["policy", "hr", "verify_node", "__end__"]:
    agent = state.get("next_agent", "finish")
    return agent if agent != "finish" else END

# =====================================================================
# 5. Build Graph (TODO)
# =====================================================================
def build_graph():
    # TODO: Create a StateGraph(SupervisorState)
    # Add nodes: "router", "policy", "hr", "verify_node"
    # Set entry_point to "router"
    # Add conditional edges from "router" using route_decision
    # Add edges from "policy" -> "verify_node" and "hr" -> "verify_node"
    # Add edge from "verify_node" -> END
    
    # TODO: Compile with SqliteSaver(":memory:") and interrupt_before=["verify_node"]
    pass

# =====================================================================
# 6. Execution (TODO)
# =====================================================================
def run_exercise():
    graph = build_graph()
    config = {"configurable": {"thread_id": "lab-supervisor-001"}}
    
    print("=== e044: Multi-Agent with HITL ===")
    
    # Session 1: Initial request (will pause before verify_node)
    print("\n--- Session 1: Request (Should Pause at HITL) ---")
    query = {"messages": [HumanMessage(content="Look up the remote work policy.")]}
    # TODO: Stream the graph with the above query and config. Print messages as they arrive.
    
    print("\n[SYSTEM]: Paused. What is the next step the graph will execute?")
    # TODO: Print the `.next` field of the saved graph state using graph.get_state(config).next

    # Session 2: Resume after human approval
    print("\n--- Session 2: Human Approved — Resume ---")
    # TODO: Stream the graph with None as input and the SAME config. Print the final message.

if __name__ == "__main__":
    run_exercise()
