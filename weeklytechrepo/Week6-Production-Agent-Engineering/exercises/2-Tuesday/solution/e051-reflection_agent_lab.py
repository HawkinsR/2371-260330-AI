from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# =====================================================================
# 1. State Definition
# =====================================================================
class ValidationState(TypedDict):
    draft: str
    critique: list[str]
    revision_count: int
    is_perfect: bool

# =====================================================================
# YOUR TASKS
# =====================================================================

def generator_node(state: ValidationState):
    """Generates the initial draft, or revises it based on feedback."""
    critiques = state.get("critique", [])
    rev_count = state.get("revision_count", 0)
    
    print(f"\n[GENERATOR - Pass {rev_count + 1}]")
    
    # 1. If there are NO critiques, output the bad JSON draft (with single quotes).
    # Bad Draft: "{ 'name': 'test_server', 'port': 8000 }"
    # If there ARE critiques, output the good JSON draft (with double quotes).
    # Good Draft: '{ "name": "test_server", "port": 8000 }'
    if not critiques:
        draft = "{ 'name': 'test_server', 'port': 8000 }"
        print("  -> Creating Initial Draft.")
    else:
        draft = '{ "name": "test_server", "port": 8000 }'
        print("  -> Reading Critiques. Applying Revisions.")
    
    print(f"  -> Emitting Draft: {draft}")
    
    # 2. Return the updated state. Increment revision_count by 1.
    return {
        "draft": draft,
        "revision_count": rev_count + 1
    }

def evaluator_node(state: ValidationState):
    """Critiques the draft against strict rules."""
    draft = state["draft"]
    critiques = state.get("critique", [])
    
    print(f"\n[EVALUATOR] Reviewing Draft...")
    
    # 3. Rule 1 - Check for single quotes.
    # If "'" is in draft, append "Invalid JSON: Must use double quotes." to critiques.
    # Return {"critique": critiques, "is_perfect": False}
    if "'" in draft:
        feedback = "Invalid JSON: Must use double quotes."
        print(f"  -> FAILED: {feedback}")
        critiques.append(feedback)
        return {"critique": critiques, "is_perfect": False}
    
    # 4. Rule 2 - Check for the word "port".
    # If "port" is NOT in draft, append "Missing required field: port." to critiques.
    # Return {"critique": critiques, "is_perfect": False}
    if "port" not in draft:
        feedback = "Missing required field: port."
        print(f"  -> FAILED: {feedback}")
        critiques.append(feedback)
        return {"critique": critiques, "is_perfect": False}
    
    print("  -> PASSED VALIDATION. Draft is PERFECT.")
    # 5. If it passes all rules, return is_perfect True
    return {"is_perfect": True}

# =====================================================================
# ROUTING LOGIC (Do not edit)
# =====================================================================
def should_continue(state: ValidationState) -> str:
    if state.get("is_perfect"):
        print("\n[ROUTER] Validation Passed. Routing to END.")
        return "end"
        
    if state.get("revision_count", 0) >= 3:
        print("\n[ROUTER] Max loops reached! Forcing END to prevent infinite loop.")
        return "end"
        
    print("\n[ROUTER] Validation Failed. Routing back to GENERATOR.")
    return "continue"

# =====================================================================
# GRAPH COMPILATION
# =====================================================================
def build_reflection_graph():
    builder = StateGraph(ValidationState)
    
    # 6. Add the Generator and Evaluator nodes
    builder.add_node("Generator", generator_node)
    builder.add_node("Evaluator", evaluator_node)
    
    # 7. Build the execution flow:
    # START -> Generator -> Evaluator
    builder.add_edge(START, "Generator")
    builder.add_edge("Generator", "Evaluator")
    
    # 8. Add conditional edges from Evaluator using should_continue.
    # Map "continue" to "Generator", and "end" to END.
    builder.add_conditional_edges(
        "Evaluator",
        should_continue,
        {
            "continue": "Generator",
            "end": END
        }
    )
    
    return builder.compile()

# =====================================================================
# PIPELINE EXECUTION
# =====================================================================
if __name__ == "__main__":
    print("=== Agentic AI: Reflection and Validation Loop ===")
    
    try:
        graph = build_reflection_graph()
        
        initial_state = {
            "draft": "",
            "critique": [],
            "revision_count": 0,
            "is_perfect": False
        }
        
        print("Initiating Execution Loop...")
        result = graph.invoke(initial_state)
        
        print("\n" + "="*50)
        print(">>> FINAL ACCEPTED CONFIGURATION <<<")
        print(result["draft"])
        print(f"Total Iterations Required: {result['revision_count']}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"Error executing graph: {e}")
        print("Please complete the graph builder configurations.")
