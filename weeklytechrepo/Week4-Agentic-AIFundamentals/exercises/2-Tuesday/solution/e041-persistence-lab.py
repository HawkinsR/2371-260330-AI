import sqlite3
import re
from typing import Annotated, TypedDict
from langchain_aws import ChatBedrock
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]

def pii_middleware_node(state: AgentState):
    """Scans the last HumanMessage for credit card patterns and redacts them."""
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        pattern = r"\b(?:\d[ -]*?){13,16}\b"
        if re.search(pattern, last_message.content):
            print("   [SECURITY]: PII Detected — Redacting card number.")
            last_message.content = re.sub(pattern, "[REDACTED]", last_message.content)
    return {"messages": []}

def model_node(state: AgentState):
    model = ChatBedrock(
        model_id="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        region_name="us-east-1"
    )
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("middleware", pii_middleware_node)
    workflow.add_node("model", model_node)
    workflow.set_entry_point("middleware")
    workflow.add_edge("middleware", "model")
    workflow.add_edge("model", END)
    
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    memory = SqliteSaver(conn)
    return workflow.compile(checkpointer=memory)

def run_exercise():
    graph = build_graph()
    config = {"configurable": {"thread_id": "lab-session-001"}}
    
    print("=== e041: Persisting and Securing Agents ===")
    
    print("\n--- Session 1: Establishing Context (with PII) ---")
    s1 = {"messages": [HumanMessage(content="My name is Alex. My card is 4111-2222-3333-4444.")]}
    result1 = graph.invoke(s1, config)
    print(f"[AI]: {result1['messages'][-1].content}")

    print("\n--- Session 2: Resume (No context re-passed) ---")
    s2 = {"messages": [HumanMessage(content="What is my name? And what did I say about my card?")]}
    result2 = graph.invoke(s2, config)
    print(f"[AI]: {result2['messages'][-1].content}")

if __name__ == "__main__":
    run_exercise()
