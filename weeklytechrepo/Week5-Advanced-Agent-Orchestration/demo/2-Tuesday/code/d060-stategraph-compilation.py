"""
Demo: StateGraph Compilation and Routing
This script demonstrates the creation of a rigid, custom agent workflow using LangGraph's 
StateGraph, highlighting the difference between linear execution and conditional routing.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# =====================================================================
# 1. State Definition
# =====================================================================
class AgentState(TypedDict):
    """
    The Global Memory of the graph. Every node receives this dictionary,
    and whatever dictionary the node returns will update this global state.
    This acts as the single source of truth passed down the chain.
    """
    user_input: str
    category: str
    sentiment: str
    final_answer: str

# =====================================================================
# 2. Node Definitions (The Workers)
# =====================================================================
# Nodes do the actual work. They receive the current state, modify it, and return the diff.
def categorize_node(state: AgentState):
    """
    Simulates an LLM analyzing the input to determine routing path.
    """
    print(f"\n[Node: Categorizer] Analyzing input: '{state['user_input']}'")
    input_text = state["user_input"].lower()
    
    # Simple keyword heuristic instead of calling a real LLM for demonstration speed
    if "invoice" in input_text or "price" in input_text:
        cat = "billing"     # Set internal label
    else:
        cat = "technical"   # Set internal label
        
    # Detect urgency/sentiment by scanning for keywords
    sent = "urgent" if "urgent" in input_text or "asap" in input_text else "normal"
        
    print(f"  -> Determined Category: '{cat}' | Sentiment: '{sent}'")
    
    # Return JUST the diff. StateGraph automatically merges this into the global state for us.
    return {"category": cat, "sentiment": sent}

def billing_node(state: AgentState):
    """
    Handles payment-related queries. Only runs if explicitly routed here.
    """
    print(f"\n[Node: Billing] Processing billing request...")
    answer = "Your last invoice was $45.00. You can view it on your dashboard."
    
    # Append urgency if needed by reading the global state
    if state.get("sentiment") == "urgent":
        answer = "[ESCALATED] " + answer + " I have flagged this for immediate processing."
        
    return {"final_answer": answer}

def technical_node(state: AgentState):
    """
    Handles IT-related queries. Only runs if explicitly routed here.
    """
    print(f"\n[Node: Technical] Processing IT support request...")
    answer = "Please restart your router and clear your browser cache."
    return {"final_answer": answer}

def manager_escalation_node(state: AgentState):
    """
    Handles extremely furious users regardless of department.
    """
    print(f"\n[Node: Escalation] Alerting human manager!")
    return {"final_answer": "We are so sorry. A human manager will call you at this number immediately."}

# =====================================================================
# 3. Edge Definitions (The Routers)
# =====================================================================
# Edges are the pipes connecting Nodes. Conditional Edges run Python logic to decide the next pipe.
def route_by_category(state: AgentState) -> str:
    """
    A conditional edge function. It looks at the state and returns a string 
    matching the exact name of the node the graph should execute next.
    """
    print("\n  [Router] Evaluating State conditions...")
    
    # Priority Override: Always intercept purely based on sentiment
    if state.get("sentiment") == "urgent":
        print("  -> Escalating due to 'urgent' sentiment!")
        return "manager_escalation"
        
    # Standard DB Routing: Check the category the Categorizer Node saved
    category = state.get("category")
    if category == "billing":
        print("  -> Routing to Billing Department.")
        return "billing_department"
    else:
        print("  -> Routing to Technical Support.")
        return "it_department"

# =====================================================================
# 4. Graph Architecture and Compilation
# =====================================================================
def build_custom_graph():
    print("--- Compiling Custom StateGraph Architecture ---")
    
    # Initialize the architecture blueprint and attach the schema
    builder = StateGraph(AgentState)
    
    # Add Nodes (Registering the Python functions as actionable steps)
    builder.add_node("categorizer", categorize_node)
    builder.add_node("billing_department", billing_node)
    builder.add_node("it_department", technical_node)
    builder.add_node("manager_escalation", manager_escalation_node)
    
    # Define execution flow map
    
    # 1. Start ALWAYS goes to categorizer
    builder.add_edge(START, "categorizer")
    
    # 2. Categorizer branches conditionally based on our 'route_by_category' custom python logic
    builder.add_conditional_edges(
        "categorizer", # The node making the decision
        route_by_category, # The function executing the decision
        { # The Dictionary mapping the strings returned by 'route_by_category' to real Node Names
            "billing_department": "billing_department",
            "it_department": "it_department",
            "manager_escalation": "manager_escalation"
        }
    )
    
    # 3. All endpoint nodes must finish the graph by pointing to END
    builder.add_edge("billing_department", END)
    builder.add_edge("it_department", END)
    builder.add_edge("manager_escalation", END)
    
    # Lock the architecture into an executable runtime object capable of processing state
    return builder.compile()

# =====================================================================
# 5. Execution
# =====================================================================
def run_scenarios():
    # Build the rigid agent structure once
    graph = build_custom_graph()
    
    # Scenario A: Standard Technical Route
    print("\n\n=== SCENARIO A: Standard Technical Request ===")
    initial_state = {"user_input": "My wifi is broken.", "category": "", "sentiment": "", "final_answer": ""}
    # Invoke forces the data through the graph from START to END
    result_a = graph.invoke(initial_state)
    print(f"\n[Final Output]: {result_a['final_answer']}")
    
    # Scenario B: Standard Billing Route
    print("\n\n=== SCENARIO B: Standard Billing Request ===")
    initial_state = {"user_input": "Where is my invoice?", "category": "", "sentiment": "", "final_answer": ""}
    result_b = graph.invoke(initial_state)
    print(f"\n[Final Output]: {result_b['final_answer']}")
    
    # Scenario C: Priority Override Route
    print("\n\n=== SCENARIO C: URGENT Billing Request ===")
    initial_state = {"user_input": "URGENT my invoice is $50,000 wrong call me ASAP!", "category": "", "sentiment": "", "final_answer": ""}
    # The categorizer triggers 'billing', but the conditional edge sees 'urgent' and intercepts it to the manager
    result_c = graph.invoke(initial_state)
    print(f"\n[Final Output]: {result_c['final_answer']}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    run_scenarios()
