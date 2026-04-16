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
    # In a real environment, this might trigger a Vector DB Search.
    # For this exercise, use an LLM with a specialized "Researcher" prompt.
    prompt = f"Perform deep analysis on this topic: {state['messages'][-1]}"
    response = llm.invoke(prompt)
    return {"messages": [f"Researcher Results: {response.content}"]}

def orchestrator_node(state: AgentState):
    query = state['messages'][-1]
    # TODO: 1. Command the LLM to decide if research is needed (yes/no based format)
    
    decision = "yes" # Replace with LLM invoke logic evaluating `query`
    
    # TODO: 2. Return a Command utilizing `goto` targeting either the worker or END
    if "yes" in decision.lower():
        return Command(goto="research_worker_node")
    else:
        return Command(goto=END)

# TODO: 3. Construct and Compile Graph
builder = StateGraph(AgentState)

if __name__ == "__main__":
    # Test queries
    q1 = "Hi, how are you today?"
    q2 = "I need deep research on quantum computing advancements."
    
    # graph.stream({"messages": [q1]})
    # graph.stream({"messages": [q2]})
