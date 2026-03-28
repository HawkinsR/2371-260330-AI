from typing import TypedDict

# =====================================================================
# 1. State Definition
# =====================================================================
class ContentState(TypedDict):
    task: str
    research_notes: str
    final_article: str
    next_action: str

# =====================================================================
# YOUR TASKS
# =====================================================================

def researcher_agent_node(state: ContentState):
    """Specialized in gathering facts and sources."""
    print(f"\n[Researcher Agent] Analyzing task: '{state['task']}'")
    
    # 1. TODO: Return a dictionary updating "research_notes"
    # Content should be: "Found 3 verified sources regarding AI enterprise adoption."
    
    return {}

def editor_agent_node(state: ContentState):
    """Specialized in writing compelling articles based on notes."""
    print(f"\n[Editor Agent] Reviewing notes: '{state.get('research_notes', '')}'")
    
    # 2. TODO: Return a dictionary updating "final_article"
    # Content should be: "AI is transforming enterprise software development."
    
    return {}

def supervisor_node(state: ContentState):
    """The Orchestrator evaluating the state and deciding the next agent."""
    print("\n[SUPERVISOR] Evaluating global state...")
    
    # 3. TODO: Implement routing logic
    # - If research_notes is empty -> Route to "Researcher"
    # - If research_notes exists, but final_article is empty -> Route to "Editor"
    # - If both exist -> Route to "FINISH"
    
    
    return {"next_action": ""}


# =====================================================================
# PIPELINE SIMULATION (Do not edit)
# =====================================================================
def run_content_pipeline():
    print("=== Multi-Agent Orchestrator Pipeline ===")
    
    state: ContentState = {
        "task": "Write an article about AI enterprise adoption.",
        "research_notes": "",
        "final_article": "",
        "next_action": ""
    }
    
    loop_count = 0
    max_loops = 5
    is_running = True
    
    while is_running and loop_count < max_loops:
        loop_count += 1
        print(f"\n--- Output Loop {loop_count} ---")
        
        # Supervisor updates the "next_action" in the state
        state.update(supervisor_node(state))
        
        action = state.get("next_action")
        print(f"  -> Supervisor Decision: {action}")
        
        if action == "FINISH":
            is_running = False
        elif action == "Researcher":
            state.update(researcher_agent_node(state))
        elif action == "Editor":
            state.update(editor_agent_node(state))
        else:
            print("ERROR: Unknown action returned by Supervisor!")
            break
            
    print("\n" + "="*50)
    print(">>> MULTI-AGENT PIPELINE COMPLETE <<<")
    print(f"Notes: {state.get('research_notes')}")
    print(f"Article: {state.get('final_article')}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_content_pipeline()
