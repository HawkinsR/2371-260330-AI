import time
import uuid
import re
from typing import Dict, List, Any

# =====================================================================
# 1. Custom Middleware (PII Masking)
# =====================================================================
def pii_masking_middleware(input_text: str) -> str:
    """
    Intercepts the user input and masks sensitive data.
    """
    # Regex to find credit card patterns (simplified 16 digits)
    card_pattern = r"\d{4}-\d{4}-\d{4}-\d{4}"
    
    if re.search(card_pattern, input_text):
        print("   [SECURITY ALERT]: Detected sensitive Credit Card information! Masking...")
        masked_text = re.sub(card_pattern, "[REDACTED_CARD]", input_text)
        return masked_text
    
    return input_text

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
        print(f"   [CHECKPOINTER]: Saving state for thread '{thread_id}'...")
        # Deep copy simulation via dictionary storage
        self.storage[thread_id] = {
            "messages": list(state["messages"]),
            "steps": state["steps"]
        }

    def load(self, thread_id: str) -> Any:
        print(f"   [CHECKPOINTER]: Loading state for thread '{thread_id}'...")
        return self.storage.get(thread_id, {"messages": [], "steps": 0})

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
    ai_response = f"I have processed your request: '{safe_query[:20]}...'"
    state["messages"].append({"role": "assistant", "content": ai_response})
    state["steps"] += 1

    # 4. Save State
    db.save(thread_id, state)
    print(f"   AI Output: {ai_response}")

if __name__ == "__main__":
    print("=== Week 4: Middleware & Persistence Lab ===")
    
    database = MockCheckpointer()
    my_thread = "session_123"

    # Scenario 1: Initial Interaction (Sensitive data)
    print("\n--- Session 1: Initial Query (Sensitive) ---")
    query_1 = "My balance for card 4111-2222-3333-4444?"
    run_persistent_agent(query_1, my_thread, database)

    # Scenario 2: Simulate System Restart & Resumption
    print("\n" + "="*40)
    print("--- SYSTEM RESTARTED (RAM CLEARED) ---")
    print("="*40)
    
    # Even though 'state' variable is lost, we can resume using 'my_thread'
    print("\n--- Session 2: Resuming Conversation ---")
    query_2 = "Also show my last transaction."
    run_persistent_agent(query_2, my_thread, database)

    # Final State Verification
    final_history = database.load(my_thread)
    print("\n>>> FINAL PERSISTED HISTORY <<<")
    for msg in final_history["messages"]:
        print(f"[{msg['role'].upper()}]: {msg['content']}")
