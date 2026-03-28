"""
Demo: Async Operations and Streaming
This script demonstrates the difference between slow synchronous blocking 
execution and fast asynchronous streaming execution using LangGraph's 
'astream_events' API.
"""

import asyncio
import time
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# =====================================================================
# 1. State Definition
# =====================================================================
class AsyncState(TypedDict):
    """The local state managed by the StateGraph."""
    query: str
    tool_data: str
    final_answer: str

# =====================================================================
# 2. Asynchronous Nodes simulating slow network I/O
# =====================================================================
# Note the 'async def' which marks this as a coroutine.
async def async_tool_node(state: AsyncState):
    """
    Simulates a tool that takes 2.5 seconds to fetch data from a database.
    Notice the 'await asyncio.sleep(2.5)'. This yields control back to the 
    event loop, allowing the server to handle other users while waiting!
    """
    print("\n   [Server] Thread executing Database Tool (takes 2.5s)...")
    await asyncio.sleep(2.5) # Non-blocking wait
    return {"tool_data": "Found 5 records matching the query."}

async def async_llm_node(state: AsyncState):
    """
    Simulates an LLM synthesizing a response over time.
    In real streaming, the LLM yields words one chunk at a time.
    """
    print("   [Server] Thread starting LLM generation...")
    # We will simulate the chunk generation logic down in the streaming handler
    await asyncio.sleep(1.0) # Simulate initial latency
    return {"final_answer": "Based on my research, there are 5 records matching your request."}

# =====================================================================
# 3. Graph Construction
# =====================================================================
def build_async_graph():
    """Builds a simple, linear StateGraph."""
    builder = StateGraph(AsyncState)
    builder.add_node("DatabaseTool", async_tool_node)
    builder.add_node("LLMSynthesis", async_llm_node)
    
    builder.add_edge(START, "DatabaseTool")
    builder.add_edge("DatabaseTool", "LLMSynthesis")
    builder.add_edge("LLMSynthesis", END)
    
    # Compile the graph into an executable format
    return builder.compile()

# =====================================================================
# 4. Execution Patterns (Blocking vs Streaming)
# =====================================================================

async def simulate_synchronous_blocking_request(graph):
    """
    How a traditional Flask/Django app handles requests. 
    The user stares at a spinner for 4 seconds until the *entire* graph finishes.
    """
    print("\n--- SYNCHRONOUS (BLOCKING) INVOCATION ---")
    print("User clicked 'Submit'. Waiting for the entire graph to finish...")
    start_time = time.time()
    
    # Execution freezes here on the client side
    # 'ainvoke' is the asynchronous version of 'invoke'
    result = await graph.ainvoke({"query": "Find records"})
    
    end_time = time.time()
    print(f"\n[Client Browser] Received JSON Response exactly {end_time - start_time:.2f} seconds later.")
    print(f"[Client Browser] Final Content: '{result['final_answer']}'")


async def simulate_asynchronous_streaming_request(graph):
    """
    How modern Next.js/React apps handle LLM requests using WebSockets or SSE (Server-Sent Events).
    The user gets live updates about what the agent is currently thinking/doing.
    """
    print("\n" + "="*50)
    print("--- ASYNCHRONOUS (STREAMING) INVOCATION ---")
    print("User clicked 'Submit'. Listening for live events...")
    start_time = time.time()
    
    # We open a persistent connection to the graph using 'astream_events'
    # 'version="v2"' is required by the LangChain API for newer event formats
    events = graph.astream_events(
        {"query": "Find records"},
        version="v2"
    )
    
    print("\n[Client Browser] Live UI Updates:")
    
    # As the graph executes nodes remotely, it fires events *immediately* to us locally.
    # We iterate over this async generator to process events as they arrive.
    async for event in events:
        event_name = event["event"]
        node_name = event["name"]
        
        # 'on_chain_end' means a node (or the whole graph) finished running
        if event_name == "on_chain_end":
            if node_name == "DatabaseTool":
                # UI Update: Spinning icon turns into a checkmark the moment the DB returns
                elapsed = time.time() - start_time
                print(f"  [{elapsed:.1f}s] ✓ Database query completed.")
                
            elif node_name == "LLMSynthesis":
                # UI Update: Mark completion
                elapsed = time.time() - start_time
                print(f"\n  [{elapsed:.1f}s] ✓ LLM generation fully completed.")
                
        # 'on_chain_start' means a node just started executing
        # Simulating live typing chunks (if we were using a real LangChain ChatModel with streaming=True)
        # We manually simulate the UX of receiving tokens here.
        if node_name == "LLMSynthesis" and event_name == "on_chain_start":
             elapsed = time.time() - start_time
             print(f"  [{elapsed:.1f}s] ✍️  Agent Typing: ", end="", flush=True)
             
             # Fake streaming loop representing the LLM token payload arriving chunk-by-chunk
             words = "Based on my research, there are 5 records matching your request.".split()
             for word in words:
                 await asyncio.sleep(0.3) # Wait 300ms between tokens to simulate slow LLM generation
                 print(word + " ", end="", flush=True)
             print() # new line
             
    end_time = time.time()
    print(f"\n[Client Browser] Entire process finished in {end_time - start_time:.2f} seconds.")
    print("="*50 + "\n")

# =====================================================================
# Main Loop Setup
# =====================================================================
async def main():
    """The main entry point for our async application."""
    graph = build_async_graph()
    
    # Run the bad way (User waits forever)
    await simulate_synchronous_blocking_request(graph)
    
    # Run the modern way (Live updates)
    await simulate_asynchronous_streaming_request(graph)

if __name__ == "__main__":
    # Standard Python architecture for launching asynchronous code
    # 'asyncio.run' starts the event loop and executes our 'main' coroutine
    asyncio.run(main())
