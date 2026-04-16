import operator
from typing import Annotated, TypedDict
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

class AgentState(TypedDict):
    messages: Annotated[list[str], operator.add]

llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0", region_name="us-east-1")

def research_worker_node(state: AgentState):
    print("--> Handed Off to Research Worker")
    prompt = f"Perform deep analysis on this topic: {state['messages'][-1]}"
    response = llm.invoke(prompt)
    return {"messages": [f"Researcher Results: {response.content}"]}

def orchestrator_node(state: AgentState):
    query = state['messages'][-1]
    
    # SOLUTION 1: Dynamically evaluate intent via Bedrock LLM
    eval_prompt = f"Determine if this query requires deep research. Reply with only 'YES' or 'NO': {query}"
    decision = llm.invoke(eval_prompt).content.strip()
    
    print(f"Orchestrator Decision: Requires Research? -> {decision}")
    
    # SOLUTION 2: Use LangGraph Command API to route dynamically
    if "yes" in decision.lower() or "true" in decision.lower():
        return Command(goto="research_worker_node")
    else:
        return Command(goto=END)

# SOLUTION 3: Construct and Compile Graph
builder = StateGraph(AgentState)
builder.add_node("orchestrator", orchestrator_node)
builder.add_node("research_worker_node", research_worker_node)

builder.add_edge(START, "orchestrator")
builder.add_edge("research_worker_node", END)

graph = builder.compile()

if __name__ == "__main__":
    print("=== Exercise 047: Multi-Agent Handoffs ===")
    
    q1 = "Hi, how are you today?"
    print(f"\nEvaluating: '{q1}'")
    for event in graph.stream({"messages": [q1]}):
        print(event)
        
    q2 = "I need deep research on quantum computing advancements."
    print(f"\nEvaluating: '{q2}'")
    for event in graph.stream({"messages": [q2]}):
        print(event)
