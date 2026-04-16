import operator
from typing import Annotated, TypedDict
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
# TODO: 1. Import interrupt from langgraph.types

class AgentState(TypedDict):
    messages: Annotated[list[str], operator.add]

llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0", region_name="us-east-1")

def vector_search_node(state: AgentState):
    print("\n--- NODE: Vector DB Search ---")
    # TODO: 2. Add an interrupt checkpoint here, prompting for user validation 
    # before executing the dummy search.
    
    return {"messages": ["System retrieved safe internal policy details."]}

def router_node(state: AgentState):
    # TODO: 3. Ask Bedrock if the user's latest message belongs in 'vector_search' or 'web_search'.
    query = state['messages'][-1]
    
    decision = "vector_search" # Replace this hardcode with LLM boolean / category evaluation
    
    print(f"Router Decision: {decision}")
    
    from langgraph.types import Command
    if "vector" in decision:
        return Command(goto="vector_search")
    else:
        return Command(goto=END)

# TODO: 4. Build graph, attach a Checkpointer memory block, and compile.
builder = StateGraph(AgentState)

if __name__ == "__main__":
    print("=== Exercise 048: Adaptive RAG Starter ===")
    # config = {"configurable": {"thread_id": "lab_user_1"}}
    
    # for event in graph.stream({"messages": ["What is our holiday policy?"]}, config):
    #     print(event)
