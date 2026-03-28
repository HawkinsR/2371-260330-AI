# Lab: Tracing Agent Execution and Context Engineering

## The Scenario
Your company's customer support AI is returning generic answers because it lacks context about who is asking the question. Furthermore, the engineering team has no idea how much money the bot is costing in API calls because there is no observability. You have been tasked with enabling LangSmith telemetry and upgrading the bot's core logic to utilize dynamic Prompt Templates that inject the user's name and subscription tier directly into the system instructions before execution.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e041-tracing_lab.py`.
3. Complete the `configure_telemetry` function:
   - Use `os.environ` to set `LANGCHAIN_TRACING_V2` to `"true"`.
   - Set the `LANGCHAIN_PROJECT` to `"Customer-Support-Bot-V1"`.
   - Set the `LANGCHAIN_ENDPOINT` to `"https://api.smith.langchain.com"`.
4. Complete the `compile_support_prompt` function:
   - Create a multiline string for the `system_template` that includes three placeholders: `{name}`, `{tier}`, and `{issue}`.
   - The prompt should instruct the AI to be a helpful support agent addressing the user by their `{name}`. 
   - Add conditional instructions: If `{tier}` is "Premium", provide priority support context.
   - Format the template string using the provided arguments and return the final assembled string.

## Definition of Done
- The script executes successfully.
- It prints that telemetry has been activated via environment variables.
- The console prints the dynamically assembled prompt showing the injected `name`, `tier`, and `issue` variables accurately.
- The simulated execution log outputs the mock LangSmith trace.
