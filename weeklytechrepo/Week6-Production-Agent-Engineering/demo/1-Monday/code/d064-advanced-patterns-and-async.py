import asyncio
from typing import Annotated, TypedDict
import operator

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
# SOLUTION: Incorporate Store Interface
from langgraph.store.memory import InMemoryStore
# SOLUTION: Incorporate Checkpointers for session retention
from langgraph.checkpoint.memory import MemorySaver

class AgentState(TypedDict):
    messages: Annotated[list[str], operator.add]
    plan: list[str]
    current_step: str

llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0", region_name="us-east-1")

async def planner_node(state: AgentState, store: InMemoryStore):
    # Retrieve user preference from the Store namespace instead of local state thread.
    pref = store.get(("user_preferences", "user_123"), "verbosity")
    style = pref.value if pref else "concise"

    print(f"\n--- PLANNER NODE [User Pref: {style}] ---")
    query = state['messages'][-1]
    
    prompt = f"Create a two-step plan to solve this: {query}. Keep the steps {style}."
    # SOLUTION: Use async .ainvoke
    response = await llm.ainvoke(prompt)
    
    # Store the generated plan
    steps = [s.strip() for s in response.content.split('\n') if s.strip()]
    return {"plan": steps}

async def executor_node(state: AgentState):
    print("--- EXECUTOR NODE ---")
    if not state.get("plan"):
        return {"messages": ["Execution complete."]}
    
    # Pop the next step
    current_plan = state["plan"]
    next_step = current_plan[0]
    
    print(f"Executing: {next_step}")
    
    # Evaluate step via LLM
    response = await llm.ainvoke(f"Execute this step: {next_step}")
    
    return {
        "messages": [f"Result of {next_step}: {response.content}"], 
        "plan": current_plan[1:]
    }

def route_execution(state: AgentState):
    if len(state.get("plan", [])) == 0:
        return END
    return "executor"

# Build Graph
builder = StateGraph(AgentState)
builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
# Routing iteration back upon itself until plan array is empty
builder.add_conditional_edges("executor", route_execution)

memory = MemorySaver()
store = InMemoryStore()

# Compile the graph binding both Checkpointers and global Store
graph = builder.compile(checkpointer=memory, store=store)

async def main():
    print("=== Demo 064: Advanced Patterns & Async Execution ===\n")
    
    # SOLUTION: Pre-inject user memory into the global Store
    store.put(("user_preferences", "user_123"), "verbosity", "extremely detailed and pedantic")
    
    config = {"configurable": {"thread_id": "session_888"}}
    inputs = {"messages": ["How do I deploy an AWS Lambda function?"]}
    
    # SOLUTION: Stream utilizing `.astream` for asynchronous performance
    print("Initiating async stream...")
    async for event in graph.astream(inputs, config):
        print(event)

if __name__ == "__main__":
    asyncio.run(main())
