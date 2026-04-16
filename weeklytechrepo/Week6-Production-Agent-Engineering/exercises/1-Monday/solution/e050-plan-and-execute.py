import asyncio
import operator
from typing import Annotated, TypedDict
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

class AgentState(TypedDict):
    messages: Annotated[list[str], operator.add]
    plan: list[str]

llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0", region_name="us-east-1")

async def planner_node(state: AgentState, store: InMemoryStore):
    # SOLUTION 1: Store retrieval parsing tuple namespace
    pref_item = store.get(("user_preferences", "user_123"), "verbosity")
    preference = pref_item.value if pref_item else "concise"
    
    print(f"\n[Planner Activated] Memory Store Read: Style -> {preference}")
    
    query = state['messages'][-1]
    prompt = f"Create a three-step plan to resolve: '{query}'. Instruction: Make the steps {preference}."
    
    response = await llm.ainvoke(prompt)
    steps = [line.strip() for line in response.content.split('\n') if line.strip()]
    return {"plan": steps}

def route_execution(state: AgentState):
    return END

async def main():
    print("=== Exercise 050 Solution: Advanced Patterns Mode ===\n")
    # SOLUTION 2: Initialize InMemoryStore and Put namespace items
    mem_store = InMemoryStore()
    mem_store.put(("user_preferences", "user_123"), "verbosity", "absurdly dramatic and pirate-themed")
    
    builder = StateGraph(AgentState)
    builder.add_node("planner", planner_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", END)
    
    # SOLUTION 3: Compile merging memory and global store
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory, store=mem_store)
    
    config = {"configurable": {"thread_id": "lab_50"}}
    inputs = {"messages": ["How do I fix a broken car engine?"]}
    
    async for event in graph.astream(inputs, config):
        for node_name, node_state in event.items():
            print(f"--- Completed Node: {node_name} ---")
            for step in node_state.get('plan', []):
                print(f"Step Output: {step}")

if __name__ == "__main__":
    asyncio.run(main())
