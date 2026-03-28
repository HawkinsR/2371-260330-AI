"""
Demo: Cross-Thread Memory Store
This script demonstrates how to integrate Persistent Checkpointers for short-term 
conversation history and a Global Store for long-term user preferences across threads.
"""

from typing import TypedDict, Annotated
import operator
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

# 1. Define the State
class AgentState(TypedDict):
    """The local state that is wiped clean or contained within a single conversation thread."""
    input: str
    response: str
    # 'operator.add' ensures that every new message is appended to the list, not overwritten
    messages: Annotated[list, operator.add]

# 2. Initialize Memory Components
# In production, these would be durable databases (e.g., Postgres, Redis)
# global_store remembers things ACROSS different conversations (like user preferences)
global_store = InMemoryStore()
# thread_checkpointer remembers things WITHIN a single conversation (like the last 5 messages)
thread_checkpointer = MemorySaver()

# 3. Define the Node Logic
def greeting_and_memory_node(state: AgentState, config: dict, store: InMemoryStore):
    """
    This node reads the user's input, checks the Global Store for preferences,
    and updates the conversation history. Notice it receives 'store' as an argument!
    """
    user_input = state["input"]
    # We extract the user_id securely from the configuration, not the LLM's imagination
    user_id = config["configurable"].get("user_id", "anonymous")
    
    print(f"\n[Node Execution] Processing input for user: {user_id}")
    
    # Define the namespace for this user's memory (like a folder path: /user_preferences/alice_123/)
    namespace = ("user_preferences", user_id)
    
    # ---------------------------------------------------------
    # PART A: Reading from the Global Store (Long-Term Memory)
    # ---------------------------------------------------------
    # Look in the user's folder for a file called "profile"
    user_profile = store.get(namespace, "profile")
    
    if user_profile:
        # We remember this user! Their data persists even if the thread_id changes.
        prefs = user_profile.value
        tone = prefs.get("preferred_tone", "friendly")
        print(f"-> Found existing profile. Tone preference: {tone}")
        
        # In a real app, an LLM would use these preferences to generate the response.
        # Here, we simulate the logic.
        if "change tone" in user_input.lower():
            # Very simple simulation of extracting intent
            new_tone = user_input.split("to")[-1].strip()
            
            # Update the Global Store so the agent remembers this forever
            store.put(namespace, "profile", {"preferred_tone": new_tone})
            response = f"Got it. I have updated your preferred tone to: {new_tone}."
            print(f"-> Updated profile in Global Store.")
        else:
            response = f"[{tone.upper()} TONE] Yes, I remember you. You said: '{user_input}'"
            
    else:
        # First time seeing this user!
        print("-> No profile found. Creating a new one with default preferences.")
        default_prefs = {"preferred_tone": "friendly"}
        # Save this to the global store for all future threads
        store.put(namespace, "profile", default_prefs)
        response = f"Hello! This is my first time meeting you. I've set your tone to friendly. You said: '{user_input}'"

    # ---------------------------------------------------------
    # PART B: Returning the State (Short-Term Thread Memory)
    # ---------------------------------------------------------
    # The returned dictionary updates the state. The checkpointer automatically saves this
    # to the specific thread_id, so the LLM remembers the immediate context.
    return {
        "response": response,
        "messages": [f"User: {user_input}", f"Agent: {response}"] # Append to history via operator.add
    }

def build_graph():
    """Compiles the StateGraph with BOTH memory systems attached."""
    builder = StateGraph(AgentState)
    builder.add_node("Agent", greeting_and_memory_node)
    builder.add_edge(START, "Agent")
    builder.add_edge("Agent", END)
    
    # Compile the graph, attaching BOTH the short-term checkpointer and long-term store
    return builder.compile(checkpointer=thread_checkpointer, store=global_store)

def demonstrate_memory_systems():
    print("--- Persistent Memory and Checkpointers Demo ---")
    graph = build_graph()
    
    # ====== SCENARIO 1: Thread A, First Interaction ======
    print("\n\n=== SCENARIO 1: Thread A, Message 1 ===")
    # Notice we provide BOTH a thread_id (for short term) and a user_id (for long term)
    config_a = {"configurable": {"thread_id": "thread_A", "user_id": "alice_123"}}
    
    result1 = graph.invoke({"input": "Hi, I am Alice."}, config=config_a)
    print(f"\n[Final Output]: {result1['response']}")
    
    # ====== SCENARIO 2: Thread A, Second Interaction ======
    print("\n\n=== SCENARIO 2: Thread A, Message 2 ===")
    print("(Testing short-term checkpointer memory within the same thread)")
    
    # We use the EXACT SAME thread_id. The checkpointer will remember message 1.
    result2 = graph.invoke({"input": "Please change tone to formal."}, config=config_a)
    print(f"\n[Final Output]: {result2['response']}")
    
    # Let's peek at the conversation history saved by the Checkpointer
    print("\n[Checkpointer Audit] Full Conversation History for Thread A:")
    # Both message 1 and message 2 are here!
    for msg in result2["messages"]:
        print(f"  - {msg}")
        
    # ====== SCENARIO 3: Thread B, Brand New Conversation ======
    print("\n\n=== SCENARIO 3: Thread B, New Conversation ===")
    print("(Testing long-term global store across different threads)")
    
    # Notice we use a DIFFERENT thread_id (simulating a new chat window), but the SAME user_id
    config_b = {"configurable": {"thread_id": "thread_B", "user_id": "alice_123"}}
    
    result3 = graph.invoke({"input": "Hello again from a new device."}, config=config_b)
    print(f"\n[Final Output]: {result3['response']}")
    
    # Let's peek at the conversation history saved by the Checkpointer for Thread B
    print("\n[Checkpointer Audit] Full Conversation History for Thread B:")
    print("Notice how the short-term history is EMPTY, but the global tone preference carried over!")
    # Only message 3 is here. Messages 1 and 2 belong to Thread A.
    for msg in result3["messages"]:
        print(f"  - {msg}")

    print("\n" + "-" * 50)

if __name__ == "__main__":
    demonstrate_memory_systems()
