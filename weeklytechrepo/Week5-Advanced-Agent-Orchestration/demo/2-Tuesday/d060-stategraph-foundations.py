import operator
from typing import Annotated, TypedDict
from langchain_aws import ChatBedrock
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 1. Define TypedDict State
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# 2. Define standard LLM node
def agent_node(state: AgentState):
    llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0", region_name="us-east-1")
    response = llm.invoke(state["messages"])
    # Return dictionary matching the state keys. The `operator.add` reducer appends this.
    return {"messages": [response]}

# 3. Build Graph
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

# 4. Persistence with MemorySaver
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    print("=== Demo 060: StateGraph Foundations with Memory ===")
    
    # Checkpointer requires a thread configuration to track users/sessions
    config = {"configurable": {"thread_id": "demo_user_1"}}
    
    # Session 1
    print("\n--- Sending Message 1 ---")
    inputs = {"messages": [HumanMessage(content="Hi, I'm Richard. I'm learning LangGraph.")]}
    for event in graph.stream(inputs, config, stream_mode="values"):
        if hasattr(event["messages"][-1], "pretty_print"):
            event["messages"][-1].pretty_print()
        
    # Session 2 - Notice how the model remembers the previous state without us passing history!
    print("\n--- Sending Message 2 (Testing Persistence) ---")
    inputs = {"messages": [HumanMessage(content="Do you remember my name?")]}
    for event in graph.stream(inputs, config, stream_mode="values"):
        if event["messages"][-1].type == "ai" and hasattr(event["messages"][-1], "pretty_print"):
            event["messages"][-1].pretty_print()
