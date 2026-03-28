# =====================================================================
# MOCK HELPER CLASSES (Do not edit this section)
# =====================================================================
class DummyMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content
        
    def __repr__(self):
        return f"[{self.role.upper()}]: {self.content}"

# =====================================================================
# YOUR TASKS
# =====================================================================

def trim_chat_history(chat_history: list[DummyMessage], max_history: int = 2) -> list[DummyMessage]:
    """Middleware logic to drop old conversation context while preserving instructions."""
    print(f"\n--- Middleware: Trimming chat history to last {max_history} messages ---")
    print(f"Original Chat Length: {len(chat_history)} messages.")
    
    trimmed_history = []
    
    # 1. TODO: Always retain the System Prompt at index 0
    
    
    # 2. TODO: Retain the last `max_history` messages. (Hint: Use slice notation [-max_history:])
    
    
    
    print(f"Trimmed Chat Length: {len(trimmed_history)} messages.")
    for msg in trimmed_history:
        print(f"  -> {msg}")
        
    return trimmed_history


def secure_delete_tool(account_name: str, injected_user_id: str) -> str:
    """Simulates a highly destructive action requiring strict authorization."""
    print(f"\n[Tool Executing] Attempting to delete account '{account_name}'...")
    print(f"[Tool Security] Verifying Authorization for Sender ID: {injected_user_id}")
    
    # 3. TODO: Verify if the injected_user_id exactly matches "AUTH_USER_999"
    # Return "Account securely deleted." if true, "ERROR: Unauthorized." if false.
    
    
    return ""


def execute_secure_tool_call(graph_config: dict, llm_generated_args: dict):
    """Executes the tool by merging the LLM logic with the secure runtime state."""
    
    print("\n--- Runtime Tool Execution ---")
    print(f"[Core Agent] LLM requests tool call with args: {llm_generated_args}")
    
    # 4. TODO: Extract the secure authenticated user ID from the nested graph_config dictionary
    secure_id = ""
    
    # 5. TODO: Call secure_delete_tool passing both the LLM arg and the injected secure_id
    result = ""
    
    print(f"[Tool Output]: {result}")
    return result


# =====================================================================
# PIPELINE EXECUTION
# =====================================================================
def run_pipeline():
    print("=== Agentic AI: Memory and Runtime Configurations ===")
    
    # --- Part 1: Managing Memory Context ---
    # User chatted for a long time
    long_history = [
        DummyMessage("system", "You are an admin bot. Never forget this."),
        DummyMessage("human", "Hi, I am Bob."),
        DummyMessage("ai", "Hello Bob!"),
        DummyMessage("human", "What is the weather today?"),
        DummyMessage("ai", "I am an admin bot, I don't know the weather."),
        DummyMessage("human", "Delete my account."),
    ]
    
    trim_chat_history(long_history, max_history=2)
    
    # --- Part 2: Runtime Tool Injection ---
    # The secure global configuration passed internally by LangGraph when triggered
    config = {
        "configurable": {
            "authenticated_user_id": "AUTH_USER_999",
            "environment": "production"
        }
    }
    
    # The LLM asks to delete the account
    llm_args = {
        "account_name": "bob_account_01"
    }
    
    execute_secure_tool_call(config, llm_args)

    print("\n" + "="*50)

if __name__ == "__main__":
    run_pipeline()
