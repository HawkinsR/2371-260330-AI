"""
Demo: Custom Middleware and State Persistence
This script demonstrates how to architect reliable agents using:
1. Custom Middleware for message interception (e.g., PII masking).
2. State Persistence (Checkpointers) to save and resume agent progress.
3. Step-wise execution tracking via a "Scratchpad" state.
"""

import time
import uuid
from typing import Dict, List, Any

# =====================================================================
# 1. State Persistence Simulation (Checkpointer)
# =====================================================================
class MockCheckpointer:
    """
    Simulates a database-backed checkpointer (like SQLite or Postgres).
    This allows an agent to persist its 'state' across different sessions.
    """
    def __init__(self):
        self.storage: Dict[str, Any] = {}

    def save(self, thread_id: str, state: Any):
        print(f"   [CHECKPOINTER]: Saving state for thread '{thread_id}'...")
        self.storage[thread_id] = state

    def load(self, thread_id: str) -> Any:
        print(f"   [CHECKPOINTER]: Loading state for thread '{thread_id}'...")
        return self.storage.get(thread_id, {"messages": [], "steps": 0})

# =====================================================================
# 2. Custom Middleware (PII Masking)
# =====================================================================
def pii_masking_middleware(input_text: str) -> str:
    """
    Intercepts the user input and masks sensitive data before the LLM sees it.
    Demonstrates the 'Middleware' pattern for security.
    """
    print("   [MIDDLEWARE]: Intercepting message for PII scanning...")
    # Very simple simulation of PII masking (Credit Card numbers)
    masked_text = input_text
    if "4111" in masked_text: # Dummy CC mock
        masked_text = masked_text.replace("4111", "[REDACTED_CARD]")
        print("   [SECURITY]: Masked sensitive Credit Card info!")
    return masked_text

# =====================================================================
# 3. Step-wise Agent Execution
# =====================================================================
def run_persistent_agent(user_query: str, thread_id: str, db: MockCheckpointer):
    """
    Runs an agent step, saves the state, and demonstrates 'Resumption'.
    """
    # 1. Intercept via Middleware
    safe_query = pii_masking_middleware(user_query)
    
    # 2. Load existing state
    current_state = db.load(thread_id)
    current_state["messages"].append({"role": "user", "content": safe_query})
    
    print(f"\n[AGENT THREAD: {thread_id}]")
    print(f"   Processing Step {current_state['steps'] + 1}...")
    time.sleep(1)

    # 3. Simulate reasoning and tool use
    print("   [THOUGHT]: User wants to check their balance. I'll need to query the ledger.")
    time.sleep(0.5)
    print("   [ACTION]: ledger_v1.get_balance(account_id='ACC-99')")
    
    # 4. Update and Save state
    current_state["steps"] += 1
    current_state["messages"].append({"role": "assistant", "content": "Your balance is $450.25"})
    db.save(thread_id, current_state)
    
    print("-" * 50)
    print(">>> AGENT STEP COMPLETE & PERSISTED <<<\n")

# =====================================================================
# 4. Pipeline Execution (Demonstrating Resumption)
# =====================================================================
def run_demo():
    print("=== Week 4: Middleware & Persistence Demo ===")
    
    # 1. Initialize our persistent "DB"
    database = MockCheckpointer()
    my_thread = str(uuid.uuid4())[:8]

    # 2. First Interaction (Contains PII)
    print("\n--- Session 1: Initial Query ---")
    query_1 = "My card ending in 4111 is failing. What is my balance?"
    run_persistent_agent(query_1, my_thread, database)

    # 3. Simulate System Crash or Restart
    print("\n--- SYSTEM RESTARTING... (Memory Persisted in DB) ---\n")
    time.sleep(1)

    # 4. Second Interaction (Resuming from state)
    print("--- Session 2: Resuming Conversation ---")
    query_2 = "Wait, actually, list my last 3 transactions too."
    
    # Even though we lost RAM state, the database has the 'thread_id' history
    run_persistent_agent(query_2, my_thread, database)

    # 5. Final State Review
    final_state = database.load(my_thread)
    print("\n>>> FINAL PERSISTED CONVERSATION HISTORY <<<")
    for msg in final_state["messages"]:
        print(f"[{msg['role'].upper()}]: {msg['content']}")
    print("-" * 50)

if __name__ == "__main__":
    run_demo()
