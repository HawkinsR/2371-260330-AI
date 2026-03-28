from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# =====================================================================
# 1. State Definition
# =====================================================================
# 1. Define the AgentState TypedDict with keys: user_input, loyalty_tier, and final_answer (all strings)
class AgentState(TypedDict):
    user_input: str
    loyalty_tier: str
    final_answer: str


# =====================================================================
# 2. Node Definitions
# =====================================================================
def analyze_tier_node(state: AgentState):
    print(f"\n[Node: Analyzer] Checking database for user: '{state['user_input']}'")
    
    # Simulating a database lookup based on user input details
    input_text = state["user_input"].lower()
    if "enterprise" in input_text or "gold" in input_text:
        tier = "gold"
    elif "premium" in input_text:
        tier = "silver"
    else:
        tier = "bronze"
        
    print(f"  -> Determined Tier: '{tier}'")
    return {"loyalty_tier": tier}

def standard_support_node(state: AgentState):
    print(f"\n[Node: Standard Support] Processing standard queue...")
    answer = "Thank you for contacting standard support. We will reply within 48 hours."
    return {"final_answer": answer}

def priority_support_node(state: AgentState):
    print(f"\n[Node: Priority Support] Processing VIP queue...")
    answer = "Welcome to Gold VIP Support! An agent will join this chat in 60 seconds."
    return {"final_answer": answer}

# =====================================================================
# 3. Router Function Definition
# =====================================================================
def route_by_tier(state: AgentState) -> str:
    print("\n  [Router] Evaluating State conditions...")
    
    # 2. Look at state.get("loyalty_tier"). 
    # Return "priority" if the tier is "gold". Otherwise, return "standard".
    tier = state.get("loyalty_tier")
    if tier == "gold":
        return "priority"
    return "standard"

# =====================================================================
# 4. Graph Construction
# =====================================================================
def build_support_graph():
    print("--- Compiling Support Logic Architecture ---")
    
    # 3. Initialize the StateGraph with your AgentState
    builder = StateGraph(AgentState)
    
    # 4. Add the 3 nodes: "analyzer", "priority", and "standard"
    builder.add_node("analyzer", analyze_tier_node)
    builder.add_node("priority", priority_support_node)
    builder.add_node("standard", standard_support_node)
    
    # 5. Define execution flow. Link START -> "analyzer"
    builder.add_edge(START, "analyzer")
    
    # 6. Add conditional edges from "analyzer" using route_by_tier.
    # Map the output strings "priority" and "standard" to their respective nodes.
    builder.add_conditional_edges(
        "analyzer",
        route_by_tier,
        {
            "priority": "priority",
            "standard": "standard"
        }
    )
    
    # 7. Link the terminal nodes "priority" and "standard" back to END
    builder.add_edge("priority", END)
    builder.add_edge("standard", END)
    
    # 8. Compile and return the graph
    return builder.compile()

# =====================================================================
# PIPELINE EXECUTION
# =====================================================================
if __name__ == "__main__":
    try:
        graph = build_support_graph()
        
        # Scenario A: Standard Tier
        print("\n\n=== SCENARIO A: Standard Request ===")
        initial_state_a = {"user_input": "I have an issue with my premium account.", "loyalty_tier": "", "final_answer": ""}
        result_a = graph.invoke(initial_state_a)
        print(f"\n[Final Output]: {result_a['final_answer']}")
        
        # Scenario B: Enterprise Tier
        print("\n\n=== SCENARIO B: Gold Enterprise Request ===")
        initial_state_b = {"user_input": "Our enterprise gold server is down!", "loyalty_tier": "", "final_answer": ""}
        result_b = graph.invoke(initial_state_b)
        print(f"\n[Final Output]: {result_b['final_answer']}")
        
    except Exception as e:
        print(f"Error executing graph: {e}")
        print("Please complete the required builder configurations.")
