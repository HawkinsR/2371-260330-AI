# Lab: Building a Basic ReAct Agent

## The Scenario
Your retail banking client wants an AI assistant that can calculate loan interest over time. Instead of relying on the LLM's internal math (which frequently hallucinates numbers), you need to build an agent that uses a dedicated Python function as a "Tool" to execute the precise mathematical formula. The agent must return its final answer adhering to a specific JSON schema so the bank's frontend website can render the data.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e040-basic_agent_lab.py`.
3. Complete the `tool_calculate_interest` function:
   - Calculate simple interest: `principal * rate * years`. Add the interest to the `principal` for the `total_amount`.
   - Return a `float` representing the final total amount.
4. Complete the `LoanReport` TypedDict struct:
   - Add the necessary schema fields exactly as defined in the instructions (e.g., `principal`, `rate`, `years`, `total_amount`, `summary`).
5. Complete the `simulate_react_agent` function:
   - Write a strict `system_prompt` instructing the agent to never hallucinate numbers, to always use the calculator tool, and to return the payload adhering precisely to the `LoanReport` schema.
   - Execute the steps of the ReAct loop: Reason, Act (call the tool), Observe, and return the Final Structured Output.

## Definition of Done
- The script executes successfully.
- The `tool_calculate_interest` tool accurately computes the math and logs the execution.
- The final output is printed as a valid JSON dictionary matching the `LoanReport` structure indicating that a $10,000 loan at 0.05 rate over 3 years yields a total amount of $11,500.0.
