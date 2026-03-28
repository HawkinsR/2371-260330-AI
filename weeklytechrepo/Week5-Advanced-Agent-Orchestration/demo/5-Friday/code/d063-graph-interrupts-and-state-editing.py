"""
Demo: Graph Interrupts, Breakpoints, and Human-in-the-Loop
This script demonstrates how to pause an autonomous agent natively BEFORE it executes
a dangerous action (like sending an email or issuing a refund). It shows how a human
can inspect the State, approve it, or reject/edit it via LangGraph Commands.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
# We use a simple in-memory saver for the demo. In prod, use PostgresSaver to persist state across server restarts.
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.messages import HumanMessage

# =====================================================================
# 1. State Definition
# =====================================================================
class ApprovalState(TypedDict):
    """The global memory structure holding the context of our approval workflow."""
    user_request: str   # What the user originally asked for
    draft_email: str    # The working draft the AI creates
    is_approved: bool   # The explicit boolean flag a Human sets
    status: str         # A string tracking where we are in the pipeline

# =====================================================================
# 2. Node Definitions
# =====================================================================
def draft_email_node(state: ApprovalState):
    """Simulates an LLM writing a potentially dangerous email based on limited context."""
    print(f"\n[Node: Drafter] Generating email based on: '{state['user_request']}'")
    
    # The AI hallucinates a massive, unauthorized discount because it thinks it's being helpful!
    draft = "Hello Customer, we are so sorry. Here is a 90% discount code: FREE90."
    
    print("  -> Draft generated successfully.")
    # We update the global state with the draft, but do NOT set is_approved
    return {"draft_email": draft, "status": "drafted"}

def send_email_node(state: ApprovalState):
    """Simulates an irreversible action that costs the company money or reputation."""
    print("\n[Node: Email Sender] *** DANGEROUS ACTION EXECUTING ***")
    
    # Final internal safety check
    if state.get("is_approved"):
        print(f"  -> Sending Email: '{state['draft_email']}'")
        return {"status": "sent"}
    else:
        print("  -> ERROR: Attempted to send without approval!")
        return {"status": "failed_unauthorized"}

def cancellation_node(state: ApprovalState):
    """Handles workflows that a human explicitly rejected."""
    print("\n[Node: Cancellation] Workflow rejected by supervisor.")
    return {"status": "cancelled"}

# =====================================================================
# 3. Routing
# =====================================================================
def route_approval(state: ApprovalState) -> str:
    """A conditional edge determining if we proceed to Send or Cancel."""
    # If the human approved it (via manual state injection), send it to the SendEmail node.
    if state.get("is_approved"):
         return "SendEmail"
    # Otherwise, cancel it.
    return "Cancel"

# =====================================================================
# 4. Compile Graph WITH Breakpoints
# =====================================================================
def build_interruptible_graph():
    """Compiles the StateGraph and adds a Checkpointer to enable 'pausing'."""
    builder = StateGraph(ApprovalState)
    
    # Register the worker nodes
    builder.add_node("DraftEmail", draft_email_node)
    builder.add_node("SendEmail", send_email_node)
    builder.add_node("Cancel", cancellation_node)
    
    # Define the starting point
    builder.add_edge(START, "DraftEmail")
    
    # We route dynamically based on whatever the human manually injects into the State
    builder.add_conditional_edges(
        "DraftEmail",
        route_approval,
        {"SendEmail": "SendEmail", "Cancel": "Cancel"}
    )
    
    # Close off the terminal paths
    builder.add_edge("SendEmail", END)
    builder.add_edge("Cancel", END)
    
    # Initialize the database that will store the frozen state while it waits for a human
    memory = MemorySaver()
    
    # CRITICAL: We tell the orchestrator to HALT execution the exact millisecond 
    # it attempts to enter either the SendEmail or Cancel nodes.
    # It will save the state to 'memory' and yield control back to the main thread.
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["SendEmail", "Cancel"]
    )
    return graph

# =====================================================================
# 5. Execution and Human-in-the-Loop Simulation
# =====================================================================
def run_human_in_the_loop_demo():
    print("--- LangGraph Breakpoint & Approval Demo ---")
    graph = build_interruptible_graph()
    
    # Thread ID is required for Checkpointers. It tracks THIS specific conversation.
    # Without a Thread ID, the graph wouldn't know which frozen state to wake up later.
    config = {"configurable": {"thread_id": "customer_101_refund"}}
    
    # --- PHASE 1: Autonomous Execution ---
    print("\n[System] Invoking graph. AI is running autonomously...")
    initial_input = {"user_request": "The customer is angry about a late delivery."}
    
    # The graph will run DraftEmail, hit the router, and then FREEZE before SendEmail/Cancel.
    # invoke() will return early, and the script continues.
    graph.invoke(initial_input, config=config)
    
    # --- PHASE 2: The Graph is Asleep ---
    print("\n" + "!"*50)
    print("!!! GRAPH EXECUTION PAUSED AT BREAKPOINT !!!")
    print("!"*50)
    
    # We can fetch the sleeping state from the DB using the config thread_id
    sleeping_state = graph.get_state(config)
    print(f"\n[Supervisor UI] Please review the AI's proposed action:")
    print(f"  Proposed Draft: '{sleeping_state.values.get('draft_email')}'")
    
    # --- PHASE 3: Human Intervention (Time Travel / State Editing) ---
    print("\n[Supervisor UI] The AI offered 90% off! That is too high.")
    print("[Supervisor UI] Manually editing the state to reject and rewrite...")
    
    # We inject a direct override into the Graph's memory for this specific thread.
    # This edits the "Past" before the graph wakes up.
    graph.update_state(
        config,
        # We deny approval AND we rewrite the email ourselves
        {"is_approved": False, "draft_email": "Customer, here is a 10% discount: TENOFF."}
    )
    
    # Check the state again to prove we edited reality
    woken_state = graph.get_state(config)
    print(f"  Overridden Draft: '{woken_state.values.get('draft_email')}'")
    print(f"  Approved Flag: {woken_state.values.get('is_approved')}")
    
    # --- PHASE 4: Resume Execution ---
    print("\n[System] Resuming graph execution from breakpoint...")
    # Passing 'None' tells LangGraph: "Just pick up exactly where you left off using the current state in DB"
    # It evaluates the router again, sees is_approved is False, and routes to Cancel!
    graph.invoke(None, config=config)
    
    print("\n" + "="*50)

if __name__ == "__main__":
    run_human_in_the_loop_demo()
