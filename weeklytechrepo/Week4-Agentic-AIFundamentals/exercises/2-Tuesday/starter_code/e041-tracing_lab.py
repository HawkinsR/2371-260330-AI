import os
import time

def configure_telemetry():
    """
    Simulates enabling LangSmith environment variables.
    """
    print("\n[System] Enabling Telemetry...")
    
    # 1. TODO: Enable LangChain tracing (LANGCHAIN_TRACING_V2)
    
    
    # 2. TODO: Set the LangChain project name (LANGCHAIN_PROJECT = 'Customer-Support-Bot-V1')
    
    
    # 3. TODO: Set the endpoint to 'https://api.smith.langchain.com'
    
    
    print("  -> Telemetry active! All chains will now trace to the cloud dashboard.")

def compile_support_prompt(name: str, tier: str, issue: str) -> str:
    """
    Demonstrates Context Engineering by dynamically injecting variables into a template.
    """
    # 4. TODO: Define a system template string with placeholders for name, tier, and issue.
    # e.g. "You are helping {name} who is on the {tier} plan with this issue: {issue}"
    system_template = """
    
    """
    
    # 5. TODO: Format the string using the arguments
    formatted_prompt = ""
    
    return formatted_prompt

def simulate_traced_call(prompt: str):
    """
    Simulates sending the prompt to the LLM and logging the trace.
    """
    print("\n[LangChain] Executing LLM call...")
    start_time = time.time()
    time.sleep(0.8) # Simulate network latency
    
    # Dummy response based on the assembled prompt
    response = "Hello! I see you are having trouble with your password. Let me help you reset that immediately since you are a valued customer."
    
    end_time = time.time()
    
    print("-" * 50)
    print("  [LANGSMITH MOCK TRACE]")
    print(f"  - Model: gpt-4o-mini")
    print(f"  - Latency: {end_time - start_time:.2f}s")
    print(f"  - Output Tokens: 25")
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
