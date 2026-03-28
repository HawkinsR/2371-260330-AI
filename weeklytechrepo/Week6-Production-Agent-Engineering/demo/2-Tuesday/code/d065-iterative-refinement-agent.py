"""
Demo: Iterative Refinement Agent
This script demonstrates how to build a self-correcting LangGraph agent that 
uses an Evaluator node to critique a Generator node's output, forcing 
iterative rewrites until a strict quality standard is met.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. Define the State
class ReflectionState(TypedDict):
    """The global memory tracking the life cycle of a single document."""
    task: str
    draft: str
    critique: list[str]  # A history of critiques so the Writer doesn't repeat mistakes
    revision_count: int  # A safety counter to prevent infinite loops
    is_perfect: bool     # Flag to break the loop when the Editor is satisfied

# 2. Define the Generator Node
def generator_node(state: ReflectionState):
    """
    This node acts as the 'Writer'. It looks at the task, and if there are 
    critiques present in the state, it attempts to fix them.
    """
    task = state["task"]
    critiques = state.get("critique", [])
    rev_count = state.get("revision_count", 0)
    
    print(f"\n[GENERATOR - Revision {rev_count}] Processing Task: '{task}'")
    
    if not critiques:
        # Initial Draft Generation (No feedback exists yet)
        draft = "LangGraph is a library. It builds graphs. I like it."
        print(f"-> Generated Initial Draft: '{draft}'")
    else:
        # Revision Generation based on Feedback from the Editor
        latest_feedback = critiques[-1]
        print(f"-> Received Feedback: '{latest_feedback}'")
        print("-> Applying revisions...")
        
        # Simulating an LLM reading the critique and trying harder each time
        if rev_count == 1:
            draft = "LangGraph, a powerful library by LangChain, constructs cyclical state graphs for agentic workflows."
        elif rev_count == 2:
            draft = "LangGraph constructs cyclical state graphs for agentic workflows. It elegantly manages persistent state and multi-agent coordination."
        else:
            draft = "LangGraph constructs cyclical state graphs for agentic workflows. It elegantly manages persistent state and multi-agent coordination. (Final Polish applied)."
            
        print(f"-> Generated Revised Draft: '{draft}'")
        
    return {
        "draft": draft,
        "revision_count": rev_count + 1 # Increment the safety counter
    }

# 3. Define the Evaluator Node (The Critique)
def evaluator_node(state: ReflectionState):
    """
    This node acts as the 'Editor'. It reviews the draft against strict rules.
    If it fails, it provides a critique. If it passes, it sets is_perfect to True.
    """
    draft = state["draft"]
    critiques = state.get("critique", [])
    
    print(f"\n[EVALUATOR] Reviewing Draft: '{draft}'")
    
    # Rule 1: Output must be at least 15 words
    word_count = len(draft.split())
    if word_count < 15:
        feedback = f"Draft is too short ({word_count} words). Please expand to at least 15 words explaining its features."
        print(f"-> FAILED Rule 1. Critique generated: '{feedback}'")
        critiques.append(feedback) # Add the feedback to the global memory so the Writer sees it
        return {"critique": critiques, "is_perfect": False}
        
    # Rule 2: Output must mention 'persistent state'
    if "persistent state" not in draft.lower():
        feedback = "Draft is missing critical concept: 'persistent state'. Please integrate this."
        print(f"-> FAILED Rule 2. Critique generated: '{feedback}'")
        critiques.append(feedback) # Add the feedback to the global memory
        return {"critique": critiques, "is_perfect": False}
        
    # If the draft survives all checks, we flag it as perfect
    print("-> PASSED ALL RULES. Draft is marked as PERFECT.")
    return {"is_perfect": True}

# 4. Define the Conditional Routing Logic
def should_continue(state: ReflectionState) -> str:
    """
    Determines whether to loop back to the Generator or exit the program entirely.
    """
    if state.get("is_perfect"):
        # The Editor loved it. We are done.
        print("\n[ROUTER] State is Perfect. Routing to END.")
        return "end" # Map to END
        
    if state.get("revision_count", 0) >= 3:
        # The Editor hates it, but we are out of time/money. Abort.
        print("\n[ROUTER] Maximum loops reached (3). Forcing END to prevent infinite loops.")
        return "end"
        
    # The Editor found issues, and we have retries left. Send it back to the Writer!
    print("\n[ROUTER] State needs work. Routing back to GENERATOR.")
    return "continue" # Map back to Generator

def build_reflection_graph():
    """Compiles the Writer-Editor loop."""
    builder = StateGraph(ReflectionState)
    
    # Add Nodes
    builder.add_node("Generator", generator_node)
    builder.add_node("Evaluator", evaluator_node)
    
    # Add Edges
    builder.add_edge(START, "Generator")
    
    # The Generator ALWAYS sends its work to the Evaluator (Editor)
    builder.add_edge("Generator", "Evaluator")
    
    # Add Conditional Edges
    # It is the Evaluator's output that determines if the cycle repeats or ends
    builder.add_conditional_edges(
        "Evaluator",
        should_continue, # Our python router logic
        {
            "continue": "Generator", # Loop back
            "end": END               # Break out
        }
    )
    
    return builder.compile()

def demonstrate_self_correction():
    print("--- Iterative Refinement Agent Demo ---")
    
    graph = build_reflection_graph()
    
    # Initialize the pristine state
    initial_state = {
        "task": "Write a comprehensive summary of LangGraph's core capabilities.",
        "critique": [],
        "revision_count": 0,
        "is_perfect": False,
        "draft": ""
    }
    
    # Run the graph in a continuous loop until someone explicitly routes to END
    print("Initiating Execution Graph...")
    final_output = graph.invoke(initial_state)
    
    print("\n" + "="*50)
    print(">>> FINAL ACCEPTED OUTPUT <<<")
    print("="*50)
    print(f"Total Revisions Attempted: {final_output['revision_count']}")
    print(f"Final Text Published: {final_output['draft']}")
    print(f"History of Critiques Received: {final_output['critique']}")
    print("-" * 50)

if __name__ == "__main__":
    demonstrate_self_correction()
