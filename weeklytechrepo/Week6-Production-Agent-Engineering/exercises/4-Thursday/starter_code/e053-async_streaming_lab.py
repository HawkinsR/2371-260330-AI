import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# =====================================================================
# 1. State Definition
# =====================================================================
class AsyncState(TypedDict):
    query: str
    report: str

# =====================================================================
# YOUR TASKS
# =====================================================================

# 1. TODO: Fix the syntax error. Make this function yield control back to the 
# event loop while it sleeps. (Hint: Use 'await')
async def analysis_node(state: AsyncState):
    print("\n   [Server] Executing deep data analysis... (This will take 2 seconds)")
    
    asyncio.sleep(2.0)  # <-- Fix this line!
    
    return {"report": "Q3 Revenue was up 15%."}


def build_graph():
    builder = StateGraph(AsyncState)
    builder.add_node("DataAnalyzer", analysis_node)
    builder.add_edge(START, "DataAnalyzer")
    builder.add_edge("DataAnalyzer", END)
    return builder.compile()


async def run_streaming_agent():
    print("=== Agentic AI: Async Streaming ===")
    graph = build_graph()
    
    print("[Client Browser] Request sent. Listening for Server-Sent Events...")
    
    # 2. TODO: Call graph.astream_events(). Pass {"query": "Analyze Q3 revenue."}
    # and require version="v2". Save to a variable named `events`.
    events = None
    
    if not events:
        print("ERROR: astream_events not implemented.")
        return

    # 3. TODO: Iterate over the events generator using `async for event in events:`
    # 4. TODO: Extract `kind = event["event"]` and `node_name = event["name"]`
    # 5. TODO: If kind is "on_chain_end" AND node_name is "DataAnalyzer", print a success message.
    
    
    
    print("\n[Client Browser] Connection closed at END of graph.")
    print("="*50 + "\n")

# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    asyncio.run(run_streaming_agent())
