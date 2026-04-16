import sqlite3
from typing import Annotated, TypedDict, Literal
from langchain_aws import ChatBedrock
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    next_agent: str

# =====================================================================
# Worker Tools
# =====================================================================
@tool
def search_policy(topic: str) -> str:
    """Search for a company policy on a given topic (e.g. 'remote work', 'PTO', 'expenses')."""
    policies = {
        "remote work": "Remote work is allowed up to 3 days/week with manager approval.",
        "pto": "Employees receive 20 days of PTO per year. Carryover limited to 5 days.",
        "expenses": "Business expenses up to $500 can be self-approved. Above that requires VP sign-off."
    }
    return policies.get(topic.lower(), f"No policy found for topic: {topic}")

@tool
def get_employee_data(emp_id: str) -> str:
    """Retrieves employee profile data for a given employee ID."""
    employees = {
        "E001": "Name: Sarah Kim | Dept: Engineering | PTO Remaining: 12 days",
        "E002": "Name: Marcus Jones | Dept: Finance | PTO Remaining: 5 days"
    }
    return employees.get(emp_id.upper(), f"No employee found with ID: {emp_id}")

# =====================================================================
# Worker Agents
# =====================================================================
llm = ChatBedrock(model_id="us.anthropic.claude-3-5-sonnet-20240620-v1:0", region_name="us-east-1")
policy_agent = create_react_agent(llm, [search_policy], state_modifier="You are a Policy Expert. Only use the search_policy tool.")
hr_agent = create_react_agent(llm, [get_employee_data], state_modifier="You are an HR Specialist. Only use the get_employee_data tool.")

# =====================================================================
# Supervisor Graph Nodes
# =====================================================================
def router_node(state: SupervisorState):
    content = state["messages"][-1].content.lower()
    if "policy" in content or "work" in content:
        return {"next_agent": "policy"}
    elif "employee" in content or "emp" in content:
        return {"next_agent": "hr"}
    return {"next_agent": "finish"}

def policy_node(state: SupervisorState):
    result = policy_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"][-1:]}

def hr_node(state: SupervisorState):
    result = hr_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"][-1:]}

def verify_node(state: SupervisorState):
    print("\n   [HITL]: Action requires human approval. Execution paused & state saved to DB.")
    return {}

def route_decision(state: SupervisorState) -> Literal["policy", "hr", "verify_node", "__end__"]:
    agent = state.get("next_agent", "finish")
    return agent if agent != "finish" else END

# =====================================================================
# Build Graph
# =====================================================================
def build_graph():
    workflow = StateGraph(SupervisorState)
    workflow.add_node("router", router_node)
    workflow.add_node("policy", policy_node)
    workflow.add_node("hr", hr_node)
    workflow.add_node("verify_node", verify_node)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges("router", route_decision)
    workflow.add_edge("policy", "verify_node")
    workflow.add_edge("hr", "verify_node")
    workflow.add_edge("verify_node", END)

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    memory = SqliteSaver(conn)
    return workflow.compile(checkpointer=memory, interrupt_before=["verify_node"])

# =====================================================================
# Execution
# =====================================================================
def run_exercise():
    graph = build_graph()
    config = {"configurable": {"thread_id": "lab-supervisor-001"}}
    
    print("=== e044: Multi-Agent with HITL ===")
    
    # Session 1: Initial request (pauses at verify_node)
    print("\n--- Session 1: Request ---")
    query = {"messages": [HumanMessage(content="Look up the remote work policy.")]}
    for chunk in graph.stream(query, config, stream_mode="values"):
        last = chunk["messages"][-1]
        print(f"[{last.type.upper()}]: {last.content}")

    saved_state = graph.get_state(config)
    print(f"\n[SYSTEM]: Paused. Next step: {saved_state.next}")

    # Session 2: Human approves → resume
    print("\n--- Session 2: Approved — Resuming ---")
    for chunk in graph.stream(None, config, stream_mode="values"):
        last = chunk["messages"][-1]
        print(f"[{last.type.upper()}]: {last.content}")

if __name__ == "__main__":
    run_exercise()
