"""
Demo: Orchestrator-Workers Architecture (Supervisor Pattern)
This script simulates a multi-agent system where a Supervisor LLM dynamically
routes tasks to specialized Worker Agents (Writer, Compliance, SEO) until
the user's overarching goal is fully achieved.
"""

from typing import TypedDict, List
import random

# =====================================================================
# 1. State Definition (Global Memory for all agents)
# =====================================================================
class MultiAgentState(TypedDict):
    """The shared whiteboard that all agents can read from and write to."""
    task: str              # The original user request guiding the whole process
    draft: str             # The current working text modified by the Writer
    compliance_check: bool # Flag indicating safety set by Compliance
    seo_keywords: List[str]# Extracted keywords set by SEO
    next_action: str       # The supervisor's routing decision dictating the next step

# =====================================================================
# 2. Worker Agents (Specialists)
# =====================================================================
def writer_agent_node(state: MultiAgentState):
    """Specialized in generating creative content. Blind to other concerns."""
    print(f"\n[Writer Agent] Received task: '{state['task']}'. Drafting content...")
    
    # In reality, this calls an LLM strictly prompted to be a writer.
    new_draft = "Artificial Intelligence is transforming enterprise software development."
    
    print("  -> Concept drafted successfully.")
    # Return just the dictionary keys we want to update in the global state
    return {"draft": new_draft}

def compliance_agent_node(state: MultiAgentState):
    """Specialized in checking text against corporate policy. Blind to SEO or creativity."""
    print(f"\n[Compliance Agent] Reviewing draft: '{state.get('draft', '')}'")
    
    # Simulating an LLM checking for PII or false claims.
    if "guarantee" in state.get('draft', '').lower() or "100%" in state.get('draft', '').lower():
        print("  -> 🚨 WARNING: Draft violates compliance! Rejecting.")
        # If it fails, we flag it so the supervisor knows it failed
        return {"compliance_check": False}
        
    print("  -> ✓ Draft passed compliance review.")
    # If it passes, we clear the flag
    return {"compliance_check": True}

def seo_agent_node(state: MultiAgentState):
    """Specialized in extracting performant keywords. Blind to compliance."""
    print(f"\n[SEO Agent] Analyzing draft for keyword optimization...")
    # Simulate an LLM extracting keywords from the draft
    keywords = ["AI", "Enterprise", "Software Development", "Future"]
    print(f"  -> Extracted keywords: {keywords}")
    # Update the global state with the found keywords
    return {"seo_keywords": keywords}

# =====================================================================
# 3. The Supervisor (The Orchestrator)
# =====================================================================
def supervisor_node(state: MultiAgentState):
    """
    Decides mathematically/semantically which agent should act next based
    on the current state of the draft and requirements. This LLM does NO writing itself.
    """
    print("\n[SUPERVISOR] Evaluating current global state...")
    
    # Sequence 1: Needs a draft first before anyone else can work
    if not state.get("draft"):
        print("  -> Decision: No draft exists. Routing to [Writer].")
        return {"next_action": "Writer"}
        
    # Sequence 2: Draft exists, but hasn't been checked for safety yet
    if state.get("compliance_check") is None:
        print("  -> Decision: Draft needs safety review. Routing to [Compliance].")
        return {"next_action": "Compliance"}
        
    # Exception Handling: Compliance failed - Rewrite!
    if state.get("compliance_check") is False:
        print("  -> Decision: Compliance failed. Routing back to [Writer] for revision.")
        # Reset compliance to None so it gets checked again AFTER the rewrite happens
        return {"next_action": "Writer", "compliance_check": None}
        
    # Sequence 3: Draft is safe, need SEO optimization
    if not state.get("seo_keywords"):
        print("  -> Decision: Draft is safe, but needs optimization. Routing to [SEO].")
        return {"next_action": "SEO"}
        
    # Sequence 4: If we made it here, the draft is written, safe, and optimized. We are done.
    print("  -> Decision: All requirements met. Routing to [FINISH].")
    return {"next_action": "FINISH"}

def supervisor_router(state: MultiAgentState) -> str:
    """The conditional edge function that maps the Supervisor's decision to actual execution pipes."""
    # This reads the 'next_action' string the supervisor generated, which dictates the Graph's next node
    return state["next_action"]

# =====================================================================
# 4. Simulation Engine
# =====================================================================
def run_orchestrator():
    print("--- Multi-Agent Orchestrator Pipeline ---")
    
    # 1. Initialize the pristine blank State
    state: MultiAgentState = {
        "task": "Write a blog about AI targeting enterprise CTOs.",
        "draft": "",
        "compliance_check": None,
        "seo_keywords": [],
        "next_action": ""
    }
    
    print(f"Original User Request: '{state['task']}'")
    
    # 2. Simulating the LangGraph Node/Edge loop dynamically
    graph_is_running = True
    loop_count = 0
    max_loops = 10 # Safety break to prevent infinite loops in bad logic
    
    while graph_is_running and loop_count < max_loops:
        loop_count += 1
        print(f"\n--- Turn {loop_count} ---")
        
        # Step A: Supervisor ALWAYS evaluates the state and decides the next move
        state.update(supervisor_node(state))
        
        # Step B: The Router evaluates the supervisor's decision to determine where to go
        next_node_name = supervisor_router(state)
        
        # Step C: Execute the elected Worker Node and merge its output into the global state
        if next_node_name == "FINISH":
            graph_is_running = False
            
        elif next_node_name == "Writer":
            state.update(writer_agent_node(state))
            
        elif next_node_name == "Compliance":
            state.update(compliance_agent_node(state))
            
        elif next_node_name == "SEO":
            state.update(seo_agent_node(state))
            
    # 3. Present Final Output securely
    print("\n" + "="*50)
    print(">>> MULTI-AGENT EXECUTION COMPLETE <<<")
    print(f"Final Draft: {state['draft']}")
    print(f"Compliance Passed: {state['compliance_check']}")
    print(f"SEO Keywords Generated: {state['seo_keywords']}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_orchestrator()
