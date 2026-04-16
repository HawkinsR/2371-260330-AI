import operator
from typing import Annotated, TypedDict
from langchain_aws import ChatBedrock
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
# Note: ToolNode and tools_condition make binding tools very simple!
from langgraph.prebuilt import ToolNode, tools_condition

# TODO: 1. Define a basic @tool for multiplication
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

tools = [multiply]

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# TODO: 2. Initialize ChatBedrock and .bind_tools(tools)
llm = None
llm_with_tools = None

def agent_node(state: AgentState):
    # TODO: 3. Invoke the LLM with the bound tools
    response = None 
    return {"messages": [response]}

# TODO: 4. Construct the Graph
builder = StateGraph(AgentState)
# builder.add_node("agent", agent_node)
# builder.add_node("tools", ToolNode(tools))

# TODO: 5. Map the logic utilizing tools_condition
# builder.add_edge(START, "agent")
# builder.add_conditional_edges("agent", tools_condition)
# builder.add_edge("tools", "agent")

if __name__ == "__main__":
    # graph = builder.compile()
    
    print("Test: What is 42 multiplied by 7?")
    # for event in graph.stream({"messages": [HumanMessage(content="What is 42 multiplied by 7?")]}):
    #     print(event)
