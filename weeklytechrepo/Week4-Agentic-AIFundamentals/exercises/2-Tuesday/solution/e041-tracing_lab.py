import os
import time

def configure_telemetry():
    """
    Simulates enabling LangSmith environment variables.
    """
    print("\n[System] Enabling Telemetry...")
    
    # 1. Enable LangChain tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    
    # 2. Set the LangChain project name
    os.environ["LANGCHAIN_PROJECT"] = "Customer-Support-Bot-V1"
    
    # 3. Set the endpoint
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    
    print("  -> Telemetry active! All chains will now trace to the cloud dashboard.")

def compile_support_prompt(name: str, tier: str, issue: str) -> str:
    """
    Demonstrates Context Engineering by dynamically injecting variables into a template.
    """
    # 4. Define a system template string
    system_template = """
    You are a helpful customer support agent.
    You are currently speaking with {name}.
    They are on the {tier} subscription plan. If they are 'Premium', give them priority service.
    
    User Issue: {issue}
    """
    
    # 5. Format the string using the arguments
    formatted_prompt = system_template.format(name=name, tier=tier, issue=issue)
    
    return formatted_prompt

def simulate_traced_call(prompt: str):
    """
    Simulates sending the prompt to the LLM and logging the trace.
    """
    print("\n[LangChain] Executing LLM call...")
    start_time = time.time()
    time.sleep(0.8) # Simulate network latency
    
    # Dummy response based on the assembled prompt
    response = "Hello Sarah! Thanks for being a Premium subscriber. I see you're having trouble with your password. Let me send a reset link right away with priority queueing."
    
    end_time = time.time()
    
    print("-" * 50)
    print("  [LANGSMITH MOCK TRACE]")
    print(f"  - Model: gpt-4o-mini")
    print(f"  - Latency: {end_time - start_time:.2f}s")
    print(f"  - Output Tokens: 32")
    print("-" * 50)
    
    return response

if __name__ == "__main__":
    print("=== Agentic AI: Context Engineering Lab ===")
    
    configure_telemetry()
    
    # Simulated DB Lookup
    user_state = {
        "name": "Sarah",
        "tier": "Premium",
        "issue": "Cannot reset password"
    }
    
    print("\nAssemble prompt with context...")
    final_prompt = compile_support_prompt(user_state["name"], user_state["tier"], user_state["issue"])
    
    print(f"\n>>> DYNAMIC PROMPT FED TO LLM <<<\n{final_prompt}")
    
    if final_prompt.strip():
        agent_reply = simulate_traced_call(final_prompt)
        print(f"\nResponse: {agent_reply}\n")
    else:
        print("\nERROR: Prompt generation failed. Please complete the compile_support_prompt function.")
