"""
Demo: Dynamic Prompts, Context Engineering, and Tracing
This script demonstrates how to construct scalable Prompt Templates using LangChain,
and how enabling LangSmith tracing automatically logs latency, token costs, 
and input/output payloads to a cloud dashboard for debugging.
"""

import os
import time

# =====================================================================
# 1. Simulating LangSmith Environment Setup
# =====================================================================
def configure_langsmith_tracing():
    """
    In a real app, this happens in your .env file or deployment pipeline.
    By merely setting these flags, the LangChain SDK transparently wraps 
    every LLM call, Tool call, and Agent step with a telemetry logger.
    """
    print("\n[System] Enabling LangSmith Telemetry...")
    
    # LANGCHAIN_TRACING_V2 is the master switch to enable tracing backend
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    # Project name organizes traces into different folders in the dashboard
    os.environ["LANGCHAIN_PROJECT"] = "Trainee-Context-Project"
    # Endpoint specifies the cloud server receiving the logs
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    # An API Key authenticates your logs. It's hidden here to prevent leaks
    # os.environ["LANGCHAIN_API_KEY"] = "ls__test_key_12345" # Hidden for demo
    
    print("  -> Telemetry active. All traces will route to the cloud dashboard.")

# =====================================================================
# 2. Context Engineering (Prompt Templates)
# =====================================================================
# We simulate the LangChain ChatPromptTemplate compiling variables
def compile_dynamic_prompt(user_name: str, subscription_tier: str, query: str) -> str:
    """
    Demonstrates Context Engineering. We do NOT just send the user's query 
    to the LLM. We wrap it in a carefully constructed System Prompt that 
    dynamically injects variables available in our application state.
    """
    print("\n[Context Engine] Compiling dynamic prompt template...")
    
    # Notice the placeholders {} which Python will replace with real variables
    system_template = """
    SYSTEM MESSAGE:
    You are a polite customer support agent. 
    The current user is named {name}. 
    They are on the {tier} subscription tier. 
    If they are on the 'Enterprise' tier, you must prioritize comprehensive technical answers.
    If they are on the 'Free' tier, you must politely remind them they can upgrade for phone support.
    """
    
    # Injecting the variables automatically into the System Payload
    formatted_system = system_template.format(name=user_name, tier=subscription_tier)
    
    # Appending the actual human question
    formatted_human = f"USER MESSAGE:\n{query}"
    
    # Combining them into one massive string block
    final_payload = formatted_system + "\n" + formatted_human
    return final_payload

# =====================================================================
# 3. Simulated Traced Execution
# =====================================================================
def simulate_traced_llm_call(formatted_prompt: str, tier: str):
    """
    Simulates sending the complex prompt to an LLM, and simultaneously 
    logging the metadata to LangSmith.
    """
    print("\n[LangChain/LangSmith] Executing traced LLM call...")
    
    # Start tracing the timer
    start_time = time.time()
    
    # Simulate network latency (Free vs Enterprise limits)
    # Usually real LLM API calls take 0.5 to 3 seconds depending on token size
    sleep_time = 0.5 if tier == "Enterprise" else 1.2
    time.sleep(sleep_time) 
    
    # Simulate the LLM's response based on the strict System Rules we injected
    if tier == "Enterprise":
         response = "Hello Alex! Let me dive into the technical details of your query regarding API limits. Our architecture..."
    else:
         response = "Hello Alex! Your data looks correct. Friendly reminder that you can upgrade to Enterprise for 24/7 phone support!"
         
    # Mark the end time
    end_time = time.time()
    latency = end_time - start_time
    
    # Simulate what LangSmith records silently in the background
    print("-" * 50)
    print("  [LANGSMITH TRACE OVERVIEW SILENTLY LOGGED TO CLOUD]")
    print(f"  - Trace ID: 4f8b92c1-xyz")
    print(f"  - Model: gpt-3.5-turbo")
    print(f"  - Latency: {latency:.2f} seconds")  # Total time from start to finish
    print(f"  - Prompt Tokens: 84")              # Number of words in the prompt
    print(f"  - Completion Tokens: 22")          # Number of words generated in response
    print(f"  - Total Cost: $0.00018")           # Calculated automatically by provider rates
    print("-" * 50)
    
    return response

# =====================================================================
# 4. Pipeline Execution
# =====================================================================
def run_context_demo():
    print("=== Agentic AI Fundamentals: Context Engineering & Tracing ===")
    
    # Turn the master switch on for tracing
    configure_langsmith_tracing()
    
    # Scenario Info retrieved safely from Database lookup
    current_user = "Alex"
    user_tier = "Free"
    user_query = "Why did my API request fail?"
    
    # 1. Format the context template using internal DB logic
    prompt = compile_dynamic_prompt(current_user, user_tier, user_query)
    print(f"\n>>> FINAL ASSEMBLED PROMPT FED TO LLM <<<\n{prompt}")
    
    # 2. Execute and Trace the assembled prompt
    agent_response = simulate_traced_llm_call(prompt, user_tier)
    
    # Show the results
    print("\n" + "="*50)
    print(">>> FINAL OUTPUT PRESENTED TO USER <<<")
    print(agent_response)
    print("="*50 + "\n")

if __name__ == "__main__":
    run_context_demo()
