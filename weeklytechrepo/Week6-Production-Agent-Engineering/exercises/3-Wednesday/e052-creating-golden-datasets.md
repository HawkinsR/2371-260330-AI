# Lab: Evaluation Driven Development (EDD)

## The Scenario
Your team is preparing to deploy an HR benefits agent to production. Before doing so, you must configure a regression test suite using Evaluation Driven Development. The agent relies on a slow `fetch_employee_benefits` tool. You must test the agent against a "Golden Dataset" containing expected answers. Crucially, you must mock the `fetch_employee_benefits` tool so the test suite runs instantly without hitting the real database.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e052-e052-evaluation_lab.py`.
3. Complete the `GOLDEN_DATASET` definition:
   - Add a second scenario dictionary to the list.
   - Set the `"input"` to `"What is the policy for remote work?"`.
   - Set the `"expected_output"` exactly to `"Employees may work from home 3 days a week."`.
4. Complete the `run_evaluation_suite` setup:
   - Utilize the `patch` context manager to mock `__main__.fetch_employee_benefits`.
   - Use the alias `as mock_tool`.
5. Complete the iteration loop:
   - Call the `hr_agent` function, passing `scenario["input"]` as the argument. Store the result.
   - Extract the `"final_answer"` string from the result.
   - Extract the `"expected_output"` string from the active `scenario`.
   - Call the `exact_match_evaluator`, passing the actual answer and expected answer. Add the returned score to `total_score`.

## Definition of Done
- The script executes successfully and instantly (because the 5-second sleep in the real tool is bypassed by your mock).
- The test suite iterates over both scenarios in the golden dataset.
- The `exact_match_evaluator` grades both scenarios.
- The console outputs an aggregate score of 100% and marks the agent as verified for production.
