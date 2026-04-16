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
    # TODO: 1. Use the injected `store` to retrieve ("user_preferences", "user_123").
    # If the user has a "verbosity" key, assign it to a local string. Otherwise default to "concise".
    preference = "concise" # Example default
    
    query = state['messages'][-1]
    # Inject that preference into the plan generation!
    prompt = f"Create a three-step plan to resolve: '{query}'. Instruction: Make the steps {preference}."
    
    # We await the async invocation
    response = await llm.ainvoke(prompt)
    steps = [line for line in response.content.split('\n') if line.strip()]
    
    return {"plan": steps}

def route_execution(state: AgentState):
    if len(state.get("plan", [])) == 0:
        return END
    return "executor"

async def main():
    # TODO: 2. Initialize an InMemoryStore and populate it.
    mem_store = InMemoryStore()
    # mem_store.put( ... )  # Insert the tuple namespace here!
    
    # TODO: 3. Compile the builder, explicitly passing checkpointer=MemorySaver() and store=mem_store
    builder = StateGraph(AgentState)
    builder.add_node("planner", planner_node)
    
    # graph = builder.compile(...)
    
if __name__ == "__main__":
    asyncio.run(main())
