import operator
from typing import Annotated, TypedDict
from langchain_aws import ChatBedrock
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# SOLUTION: 1. Define a basic @tool for multiplication
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

tools = [multiply]

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# SOLUTION: 2. Initialize ChatBedrock and bind tools
llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0", region_name="us-east-1")
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    # SOLUTION: 3. Invoke the LLM with the bound tools
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# SOLUTION: 4. Construct the Graph
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

# SOLUTION: 5. Map the logic utilizing tools_condition
builder.add_edge(START, "agent")
# tools_condition automatically routes to "tools" if a tool call is present, else END.
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()

if __name__ == "__main__":
    print("=== Exercise 046 Final Graph Stream ===\n")
    print("Test: What is 42 multiplied by 7?")
    
    inputs = {"messages": [HumanMessage(content="What is 42 multiplied by 7?")]}
    for event in graph.stream(inputs):
        for node_name, node_state in event.items():
            print(f"--- Node Executed: {node_name} ---")
            print(node_state["messages"][-1])
