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

# 1. Fix the syntax error. Make this function yield control back to the 
# event loop while it sleeps. (Hint: Use 'await')
async def analysis_node(state: AsyncState):
    print("\n   [Server] Executing deep data analysis... (This will take 2 seconds)")
    
    await asyncio.sleep(2.0)  # <-- Fixed
    
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
    
    # 2. Call graph.astream_events(). Pass {"query": "Analyze Q3 revenue."}
    # and require version="v2". Save to a variable named `events`.
    events = graph.astream_events(
        {"query": "Analyze Q3 revenue."},
        version="v2"
    )
    
    if not events:
        print("ERROR: astream_events not implemented.")
        return

    # 3. Iterate over the events generator using `async for event in events:`
    # 4. Extract `kind = event["event"]` and `node_name = event["name"]`
    # 5. If kind is "on_chain_end" AND node_name is "DataAnalyzer", print a success message.
    async for event in events:
        kind = event["event"]
        node_name = event["name"]
        
        if kind == "on_chain_end" and node_name == "DataAnalyzer":
            print("  [UI] Data Analysis Complete!")
            
    print("\n[Client Browser] Connection closed at END of graph.")
    print("="*50 + "\n")

# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    asyncio.run(run_streaming_agent())
