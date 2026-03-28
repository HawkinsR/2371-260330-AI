"""
Demo: Runtime Configuration, Message Trimming, and Tool State Injection
This script demonstrates how to manage long conversation histories to prevent context 
window explosion, and how to inject secure state variables into tools without 
relying on the LLM to provide them.
"""

# =====================================================================
# 1. Message Trimming (Managing Conversation History)
# =====================================================================
class DummyMessage:
    """A simulated LangChain chat message object."""
    def __init__(self, role, content):
        self.role = role # 'system', 'human', or 'ai'
        self.content = content # The text of the message
        
    def __repr__(self):
        return f"[{self.role.upper()}]: {self.content}"

def simulate_message_trimming():
    """Simulates how we prune old chat messages to save tokens and maintain model focus."""
    print("--- Middleware: Conversation History Trimming ---")
    
    # Imagine a user who has been chatting for 3 hours. The context window would overflow.
    long_history = [
        DummyMessage("system", "You are a highly secure bank teller bot. Never forget this."),
        DummyMessage("human", "Hi, I want to check my balance."),
        DummyMessage("ai", "Hello! Your balance is $500."),
        DummyMessage("human", "What is the weather today?"),
        DummyMessage("ai", "I'm a bank bot, I don't know the weather."),
        DummyMessage("human", "Oh right. Can I transfer $50 to Bob?"), # Most recent context matters
    ]
    
    print(f"Original Chat Length: {len(long_history)} messages.")
    
    # We want to keep the System prompt (rules), but drop the irrelevant middle conversation 
    # so we don't waste tokens and money on "What is the weather?"
    trimmed_history = []
    
    # Always keep the system instructions (Index 0) to prevent the LLM from going rogue
    trimmed_history.append(long_history[0])
    
    # Keep only the last 2 interactions (Index -2 from end) so the LLM knows what the user JUST said
    trimmed_history.extend(long_history[-2:])
    
    print(f"\nTrimmed Chat Length: {len(trimmed_history)} messages. (Saving tokens!)")
    for msg in trimmed_history:
        print(f"  -> {msg}")
    print("="*50 + "\n")


# =====================================================================
# 2. ToolRuntime and State Injection 
# =====================================================================
# Imagine the LLM decides to call this tool. 
# It passes `amount` and `recipient`. 
# BUT, it should NEVER pass the `sender_id`. That would allow the LLM to 
# hallucinate a different user's ID and steal money!
def secure_transfer_tool(amount: int, recipient: str, injected_user_id: str) -> str:
    """Simulates a tool that requires strict runtime configuration."""
    print(f"\n[Tool Executing] Attempting transfer of ${amount} to '{recipient}'...")
    print(f"[Tool Security] Verifying Authorization for Sender ID: {injected_user_id}")
    
    # Security Check: Compare the secure injected ID with our database
    if injected_user_id == "VALID_USER_777":
        return "SUCCESS: Transfer completed."
    return "ERROR: Unauthorized action."


def simulate_tool_state_injection():
    """Simulates LangChain's RunnableConfig injection pattern."""
    print("--- Runtime Configuration: Tool State Injection ---")
    
    # 1. When the user logs in and starts the graph, we configure the runtime context.
    # The LLM never sees this directly. It is stored securely in the graph execution engine's memory.
    graph_runnable_config = {
        "configurable": {
            "authenticated_user_id": "VALID_USER_777",
            "environment": "production"
        }
    }
    
    # 2. The LLM processes the user's prompt: "Can I transfer $50 to Bob?"
    # It decides to use the `secure_transfer_tool`. It ONLY generates the arguments 
    # it knows about from the prompt (Amount and Recipient).
    llm_generated_args = {
        "amount": 50,
        "recipient": "Bob"
    }
    
    print(f"[Core Agent] LLM requests tool call with args: {llm_generated_args}")
    
    # 3. Middleware intercepts the tool call BEFORE execution.
    # It dynamically injects the secure configuration variable pulled from our Graph Memory.
    secure_user_id = graph_runnable_config["configurable"]["authenticated_user_id"]
    
    # 4. The tool executes with both the LLM's logic AND the secure System State.
    result = secure_transfer_tool(
        amount=llm_generated_args["amount"], 
        recipient=llm_generated_args["recipient"],
        injected_user_id=secure_user_id
    )
    
    # 5. Return the protected output to the LLM
    print(f"\n[Tool Output]: {result}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Run the demo pipeline
    simulate_message_trimming()
    simulate_tool_state_injection()
