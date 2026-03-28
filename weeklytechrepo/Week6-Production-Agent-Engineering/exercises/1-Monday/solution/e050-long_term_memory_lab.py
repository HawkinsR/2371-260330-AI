from typing import TypedDict
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

# =====================================================================
# 1. State Definition
# =====================================================================
class AssistantState(TypedDict):
    input: str
    response: str

# =====================================================================
# YOUR TASKS
# =====================================================================

def executive_assistant_node(state: AssistantState, config: dict, store: InMemoryStore):
    """Integrates both Short-Term and Long-Term memory architectures."""
    user_input = state["input"]
    
    # 1. Extract the user_id from the graph configuration dictionary
    user_id = config["configurable"].get("user_id", "anonymous")
    
    namespace = ("executive_profiles", user_id)
    
    # 2. Look up the user's profile in the global store
    user_profile = store.get(namespace, "profile")
    
    if user_profile:
        # 3. Extract their name and style from the profile.value dict.
        # Construct a personalized response incorporating their style and input.
        name = user_profile.value.get("name", "Guest")
        style = user_profile.value.get("style", "Standard")
        
        response = f"[{style.upper()} STYLE] Good day, {name}. Regarding your request: '{user_input}'"
        print("  -> Global Profile Found! Utilizing personalized settings.")
    else:
        # 4. First time seeing this user. Store a default profile containing
        # {"name": "Guest", "style": "Standard"}.
        store.put(namespace, "profile", {"name": "Guest", "style": "Standard"})
        
        response = f"[STANDARD STYLE] Hello! I do not recognize you. I have created a default profile for you. You said: '{user_input}'"
        print("  -> First interaction. Default profile committed to Global Store.")
        
    return {"response": response}


def build_assistant_graph():
    print("--- Compiling Persistent Memory Graph ---")
    
    # 5. Instantiate BOTH the global Store and the thread Checkpointer
    store = InMemoryStore()
    checkpointer = MemorySaver()
    
    builder = StateGraph(AssistantState)
    builder.add_node("Assistant", executive_assistant_node)
    builder.add_edge(START, "Assistant")
    builder.add_edge("Assistant", END)
    
    # 6. Compile the graph, attaching the store and checkpointer
    graph = builder.compile(checkpointer=checkpointer, store=store)
    
    return graph, store # Return store for testing purposes


# =====================================================================
# PIPELINE EXECUTION (Do not edit)
# =====================================================================
def run_memory_tests():
    print("=== Agentic AI: Persistent Memory Architecture ===")
    
    try:
        graph, store = build_assistant_graph()
        
        if not graph:
            print("ERROR: Graph is not compiled. Please complete build_assistant_graph.")
            return

        # SIMULATED DATABASE UPDATE
        # Pretend an external tool updated the executive's preferences overnight
        print("\n[Admin Task] Pre-loading executive preferences into the Global Store...")
        store.put(
            ("executive_profiles", "exec_vip_007"),
            "profile",
            {"name": "Mr. Bond", "style": "Concise and Professional"}
        )

        # ====== SCENARIO 1: First Interaction (New User) ======
        print("\n\n=== SCENARIO 1: Unrecognized Executive (Thread 1) ===")
        config_1 = {"configurable": {"thread_id": "laptop_thread", "user_id": "new_hire_99"}}
        result_1 = graph.invoke({"input": "What is the status report?"}, config=config_1)
        print(f"\n[Final Output]: {result_1['response']}")

        # ====== SCENARIO 2: VIP Executive (Thread 2) ======
        print("\n\n=== SCENARIO 2: VIP Executive (Thread 2) ===")
        print("(Simulating the VIP logging in from a brand new mobile device, utilizing the global store)")
        config_2 = {"configurable": {"thread_id": "mobile_thread", "user_id": "exec_vip_007"}}
        result_2 = graph.invoke({"input": "What is the status report?"}, config=config_2)
        print(f"\n[Final Output]: {result_2['response']}")

    except Exception as e:
        print(f"Error executing memory graph: {e}")

    print("\n" + "="*50)

if __name__ == "__main__":
    run_memory_tests()
