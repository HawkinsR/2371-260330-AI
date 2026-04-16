import operator
from typing import Annotated, TypedDict, Literal
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# 1. State
class AgentState(TypedDict):
    messages: Annotated[list[str], operator.add]

# 2. LLMs
llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0", region_name="us-east-1")

# 3. Nodes
def support_agent(state: AgentState):
    prompt = f"You are a general support agent. Answer politely. Request: {state['messages'][-1]}"
    response = llm.invoke(prompt)
    return {"messages": [f"Support Specialist: {response.content}"]}

def code_agent(state: AgentState):
    prompt = f"You are a Python expert. Answer purely with code concepts. Request: {state['messages'][-1]}"
    response = llm.invoke(prompt)
    return {"messages": [f"Code Expert: {response.content}"]}

def supervisor(state: AgentState):
    # Supervisor LLM decides which agent gets the task based on semantic intent
    prompt = f"Review this user request: '{state['messages'][-1]}'. Decide if it should be routed to a 'code' expert or 'support' expert. Respond with exactly one word: 'code' or 'support'."
    decision = llm.invoke(prompt).content.strip().lower()
    
    # Dynamic routing via Command API utilizing live LLM reasoning
    if "code" in decision:
        return Command(goto="code_agent")
    else:
        return Command(goto="support_agent")

# 4. Graph Construction
builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor)
builder.add_node("code_agent", code_agent)
builder.add_node("support_agent", support_agent)

builder.add_edge(START, "supervisor")
builder.add_edge("code_agent", END)
builder.add_edge("support_agent", END)

graph = builder.compile()

if __name__ == "__main__":
    print("=== Demo 061: Multi-Agent Supervisor using Live LLM Routing ===")
    
    query1 = "How do I build a REST API in Python using FastAPI?"
    print(f"\nEvaluating: {query1}")
    for event in graph.stream({"messages": [query1]}):
        print(list(event.values())[0])

    query2 = "My order arrived damaged, where can I get a refund?"
    print(f"\nEvaluating: {query2}")
    for event in graph.stream({"messages": [query2]}):
        print(list(event.values())[0])
