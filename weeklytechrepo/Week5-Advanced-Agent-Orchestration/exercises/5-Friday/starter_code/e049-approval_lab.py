from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver 

# =====================================================================
# 1. State and Node Definitions (Do not edit this section)
# =====================================================================
class RefundState(TypedDict):
    customer_issue: str
    refund_amount: int
    is_approved: bool

def draft_refund_node(state: RefundState):
    print(f"\n[Node: Drafter] Analyzing issue: '{state['customer_issue']}'")
    # AI incorrectly decides to refund way too much!
    return {"refund_amount": 10000}

def process_refund_node(state: RefundState):
    print("\n[Node: Processor] *** DANGEROUS ACTION EXECUTING ***")
    if state.get("is_approved"):
        print(f"  -> SUCCESS: ${state['refund_amount']} refunded to customer.")
    else:
        print("  -> ERROR: Attempted to process without approval!")
    return {}

def cancel_refund_node(state: RefundState):
    print("\n[Node: Cancellation] Refund rejected by supervisor.")
    print("  -> Refund gracefully cancelled.")
    return {}

def route_refund(state: RefundState) -> str:
    if state.get("is_approved"):
        return "ProcessRefund"
    return "CancelRefund"

# =====================================================================
# YOUR TASKS
# =====================================================================

def build_interruptible_graph():
    print("--- Compiling Graph with Breakpoints ---")
    builder = StateGraph(RefundState)
    builder.add_node("DraftRefund", draft_refund_node)
    builder.add_node("ProcessRefund", process_refund_node)
    builder.add_node("CancelRefund", cancel_refund_node)
    
    builder.add_edge(START, "DraftRefund")
    builder.add_conditional_edges("DraftRefund", route_refund, {"ProcessRefund": "ProcessRefund", "CancelRefund": "CancelRefund"})
    builder.add_edge("ProcessRefund", END)
    builder.add_edge("CancelRefund", END)
    
    # 1. TODO: Initialize MemorySaver
    memory = None
    
    # 2. TODO: Compile the graph passing the checkpointer and interrupting before "ProcessRefund" and "CancelRefund"
    graph = None
    
    return graph

def run_approval_workflow():
    print("=== Agentic AI: Human-in-the-Loop Workflow ===")
    graph = build_interruptible_graph()
    
    if graph is None:
        print("ERROR: Graph is not compiled.")
        return
        
    config = {"configurable": {"thread_id": "refund_101"}}
    initial_input = {"customer_issue": "My $10 coffee was cold.", "is_approved": False}
    
    print("\n[System] Phase 1: Autonomous Execution Started...")
    graph.invoke(initial_input, config=config)
    
    print("\n" + "!"*50)
    print("!!! GRAPH EXECUTION PAUSED FOR SUPERVISOR REVIEW !!!")
    print("!"*50)
    
    # 3. TODO: Fetch the sleeping state using graph.get_state(config)
    sleeping_state = None
    
    if sleeping_state:
        proposed_amt = sleeping_state.values.get("refund_amount")
        print(f"\n[Supervisor UI] The AI proposes a refund of: ${proposed_amt}")
        print("[Supervisor UI] This is absurd. Rejecting the workflow and setting amount to 0.")
        
        # 4. TODO: Manually override the state. Set is_approved to False, and refund_amount to 0.
        
        
        
        print("\n[System] Phase 2: Resuming graph execution from breakpoint...")
        # 5. TODO: Resume graph execution by calling graph.invoke(None, config=config)
        

    print("\n" + "="*50)

if __name__ == "__main__":
    run_approval_workflow()
