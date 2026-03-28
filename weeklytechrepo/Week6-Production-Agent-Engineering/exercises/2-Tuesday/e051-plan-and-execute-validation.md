# Lab: Plan and Execute Validation

## The Scenario
Your company requires an agent that can write highly rigid JSON configurations. However, the standard LLM keeps hallucinating extra fields or formatting it incorrectly, crashing the downstream applications. You have been tasked with building an Iterative Refinement Agent. You will construct a "Generator Node" that attempts to write the configuration, and an "Evaluator Node" that critiques the draft against strict rules. The graph will loop back to the generator until the Evaluator is perfectly satisfied (or hits a loop limit).

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e051-e051-reflection_agent_lab.py`.
3. Complete the `generator_node` function:
   - Check if there are any critiques in `state.get("critique", [])`.
   - If there are NO critiques (this is the first pass), set the draft to: `"{ 'name': 'test_server', 'port': 8000 }"` (Notice this uses single quotes, which is invalid JSON!)
   - If there IS a critique, the generator tries again. Set the draft to: `'{ "name": "test_server", "port": 8000 }'` (Notice this uses double quotes, which is valid JSON).
   - Increment the `revision_count` by 1 and return the updated state.
4. Complete the `evaluator_node` function:
   - This node receives the draft. 
   - Rule 1: Validate if the draft contains double quotes (`"`). If it contains single quotes (`'`), it fails. Append a critique: `"Invalid JSON: Must use double quotes."` and return `{"is_perfect": False}`.
   - Rule 2: Validate if the draft contains the string `"port"`. If it does not, it fails. Append a critique: `"Missing required field: port."` and return `{"is_perfect": False}`.
   - If it passes both rules, return `{"is_perfect": True}` (and an empty or existing critique list).
5. Complete the `build_reflection_graph` function:
   - Add the `"Generator"` and `"Evaluator"` nodes.
   - Connect `START` to `"Generator"`.
   - Connect `"Generator"` to `"Evaluator"` (so it always evaluates after generating).
   - Add conditional edges from `"Evaluator"` using the provided `should_continue` router. Map `"continue"` to `"Generator"` and `"end"` to `END`.

## Definition of Done
- The script executes successfully.
- The console shows the initial draft failing the Evaluator's double-quote rule.
- The console shows the graph routing *back* to the Generator.
- The Generator outputs the revised draft, which then passes the Evaluator.
- The graph routes to `END` and prints the perfectly validated JSON text.
