import time
import uuid
from typing import Dict, List, Any

# =====================================================================
# 1. Custom Middleware (PII Masking)
# =====================================================================
def pii_masking_middleware(input_text: str) -> str:
    """
    Intercepts the user input and masks sensitive data.
    """
    # 1. TODO: Implement masking logic.
    # Replace anything resembling a 16-digit credit card with '[REDACTED_CARD]'.
    # Hint: Use string.replace() or a simple regex.
    masked_text = input_text
    
    # 2. TODO: Log a security alert message to the console if a card is masked.
    
    return masked_text

# =====================================================================
# 2. State Persistence (Checkpointer)
# =====================================================================
class MockCheckpointer:
    """
    Simulates a persistent database for agent state.
    """
    def __init__(self):
        self.storage: Dict[str, Any] = {}

    def save(self, thread_id: str, state: Any):
        # 3. TODO: Save the 'state' into 'self.storage' using 'thread_id' as the key.
        print(f"   [CHECKPOINTER]: Saving state for thread '{thread_id}'...")
        pass

    def load(self, thread_id: str) -> Any:
        # 4. TODO: Return the state for 'thread_id' or a default state if not found.
        # Default state: {"messages": [], "steps": 0}
        print(f"   [CHECKPOINTER]: Loading state for thread '{thread_id}'...")
        return {"messages": [], "steps": 0}

# =====================================================================
# 3. The Persistent Agent Loop
# =====================================================================
def run_persistent_agent(user_query: str, thread_id: str, db: MockCheckpointer):
    """
    Executes a single step of the agent and persists the state.
    """
    # 1. Middleware Check
    safe_query = pii_masking_middleware(user_query)
    
    # 2. Load State
    state = db.load(thread_id)
    state["messages"].append({"role": "user", "content": safe_query})
    
    # 3. Simulate AI Reasoning
    print(f"\n[AGENT THREAD: {thread_id}] Step {state['steps'] + 1}...")
    time.sleep(0.5)
    ai_response = "I have processed your request for account balance."
    state["messages"].append({"role": "assistant", "content": ai_response})
    state["steps"] += 1

    # 4. Save State
    db.save(thread_id, state)
    print(f"   AI Output: {ai_response}")

if __name__ == "__main__":
    print("=== Week 4: Middleware & Persistence Lab ===")
    
    database = MockCheckpointer()
    my_thread = "session_123"

    # Scenario: User provides sensitive info
    print("\n--- Session 1: Initial Query (Sensitive) ---")
    query_1 = "My balance for card 4111-2222-3333-4444?"
    run_persistent_agent(query_1, my_thread, database)

    # TODO: Simulate a system restart (clear local variables)
    # Then run a second query using the SAME thread_id to show persistence.
    print("\n--- Session 2: Resuming Conversation ---")
    query_2 = "Also show my last transaction."
    # run_persistent_agent(query_2, my_thread, database)
