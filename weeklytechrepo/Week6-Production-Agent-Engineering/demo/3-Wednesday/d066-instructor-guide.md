# Demo: Evaluation Driven Development

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
| --- | --- |
| **EDD vs. Manual Testing** | *"You improve your chatbot's system prompt and manually test it with 5 questions. It seems better. What specific scenarios could you be missing that an automated Golden Dataset of 500 examples would catch?"* |
| **Golden Dataset Growth** | *"A bug is reported in production: the bot gave a user the wrong PTO policy for their state. After fixing it, what EXACTLY should you do with the Golden Dataset to prevent this regression from ever silently coming back?"* |
| **LLM-as-a-Judge** | *"Why can't you use a simple Python `assert actual_output == expected_output` to evaluate whether an email response was 'polite and professional'? What does LLM-as-a-Judge provide that string comparison cannot?"* |
| **Mocking for Safety** | *"Describe what would happen if you ran 500 evaluation scenarios against a travel booking agent WITHOUT mocking the booking tool. Give at least two distinct negative consequences beyond just cost."* |


## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/evaluation-driven-development.mermaid`.
2. Trace the path from a basic Code/Prompt change to the `Offline Evaluation` block. Emphasize that in the AI era, you cannot just look at the code to know if it's right. You must evaluate the output empirically.
3. **Discussion:** Point to the `Mock out live Web/DB tools` section. Ask the class: "If we are testing an agent that books flights, and we run a test suite of 500 scenarios every night, what happens if we don't mock the tool?" (Answer: We will accidentally book 500 fake flights against the live Delta API, wasting thousands of dollars and ruining the database. Mocking is critical for offline safety).

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d066-mocking-tools-and-unit-testing.py`.
2. Review the `live_database_search_tool`. Notice the `time.sleep(1)`. This represents an expensive, slow API call.
3. Review the `GOLDEN_DATASET`. Explain that this is what a QA engineer defines as absolute truth.
4. Walk through the `run_evaluation_suite` function. 
   - Highlight the `with patch(...) as mock_tool:` context manager. Explain that this magically intercepts the Python import system. Whenever the `customer_service_agent` tries to call the real database, `unittest.mock` intercepts it and returns the fake `side_effect`.
5. Execute the script via `run_evaluation_suite()`. 
6. Watch the terminal output. Notice that `[TOOL] Connecting to Live Database` is NEVER printed. The mock worked. Notice how the Evaluation framework mathematically scores the Agent's synthesis against the Golden expected output.

## Summary
Reiterate that Evaluation Driven Development (EDD) is how you graduate from building cool weekend projects to building enterprise AI that banks and healthcare companies actually trust.
