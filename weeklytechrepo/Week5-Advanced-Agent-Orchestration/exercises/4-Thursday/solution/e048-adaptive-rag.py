import operator
from typing import Annotated, TypedDict
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

class AgentState(TypedDict):
    messages: Annotated[list[str], operator.add]

llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0", region_name="us-east-1")

def vector_search_node(state: AgentState):
    print("\n--- NODE: Vector DB Search ---")
    # SOLUTION 1: Add interrupt checkpoint
    auth = interrupt(f"Security Alert: Preparing to search internal vectors for -> '{state['messages'][-1]}'. Allow? (yes/no): ")
    
    if auth.lower() == "yes":
        return {"messages": ["System retrieved safe internal policy details."]}
    return {"messages": ["Search aborted by user override."]}

def router_node(state: AgentState):
    # SOLUTION 2: Use LLM for routing
    query = state['messages'][-1]
    prompt = f"Categorize this user query strictly as 'vector_search' (company policies) or 'web_search' (general news): {query}"
    
    decision = llm.invoke(prompt).content.strip().lower()
    print(f"Router LLM Decision: {decision}")
    
    if "vector" in decision:
        return Command(goto="vector_search")
    else:
        return Command(goto=END)

# SOLUTION 3: Compile with Checkpointer
builder = StateGraph(AgentState)
builder.add_node("router", router_node)
builder.add_node("vector_search", vector_search_node)

builder.add_edge(START, "router")
builder.add_edge("vector_search", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    print("=== Exercise 048: Adaptive RAG & HITL ===")
    config = {"configurable": {"thread_id": "lab_user_1"}}
    
    print("\n[Executing Part 1 - Will Pause at Breakpoint]")
    for event in graph.stream({"messages": ["What is our company holiday policy?"]}, config):
        print(event)
        
    print("\n[Simulating Admin Resume via Command(resume='yes')]")
    for event in graph.stream(Command(resume="yes"), config):
        print(event)
