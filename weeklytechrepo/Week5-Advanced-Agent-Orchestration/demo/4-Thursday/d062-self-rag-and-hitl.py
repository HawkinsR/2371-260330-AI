import operator
from typing import Annotated, TypedDict
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

class AgentState(TypedDict):
    messages: Annotated[list[str], operator.add]
    approved: bool

llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0", region_name="us-east-1")

def draft_email_node(state: AgentState):
    prompt = f"Draft a brief, professional response to this inquiry: {state['messages'][-1]}"
    response = llm.invoke(prompt)
    return {"messages": [f"DRAFT: {response.content}"], "approved": False}

def human_approval_node(state: AgentState):
    # Triggers a hard breakpoint in the execution graph natively via LangGraph
    decision = interrupt(f"\nACTION REQUIRED:\nReview this draft -> {state['messages'][-1]}\nApprove? (yes/no): ")
    return {"approved": True if decision.lower() == "yes" else False}

def final_send_node(state: AgentState):
    if state["approved"]:
        return {"messages": ["System: Email approved and successfully sent to downstream API."]}
    return {"messages": ["System: Email was rejected by human operator. Aborting."]}

builder = StateGraph(AgentState)
builder.add_node("drafter", draft_email_node)
builder.add_node("approver", human_approval_node)
builder.add_node("sender", final_send_node)

builder.add_edge(START, "drafter")
builder.add_edge("drafter", "approver")
builder.add_edge("approver", "sender")
builder.add_edge("sender", END)

# In order to use interrupts, a checkpointer MUST be supplied
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    print("=== Demo 062: Real Human-In-The-Loop Execution ===")
    config = {"configurable": {"thread_id": "hitl_thread_1"}}
    
    print("\n--- Starting Thread ---")
    inputs = {"messages": ["Customer: I urgently need an extension on my assignment!"]}
    
    # 1. Stream runs until the interrupt() is hit.
    for event in graph.stream(inputs, config):
        print(event)
        
    print("\n[Execution Paused at Breakpoint waiting for human...]")
    
    # 2. In a real application, you'd resume asynchronously when a user clicks 'Approve' in a UI.
    # Here, we resume by passing the 'interrupt' value explicitly via Command.
    print("Simulating User clicking 'Yes' in UI...")
    from langgraph.types import Command
    for event in graph.stream(Command(resume="yes"), config):
        print(event)
