# e051: Testing Your Agent

## Objective
Apply Evaluation Driven Development (EDD) by writing strict structural unit tests for an Agent. You will use Python's `unittest.mock.patch` library to intercept an external network tool and ensure logic loops are evaluated efficiently offline.

## Instructions
1. Open `starter_code/e051-testing-agent.py`.
2. Review the basic graph network provided. It binds an external dummy tool (`query_financial_database`).
3. In the `TestAgentFeatures` test case, import and apply the `@patch` decorator to target the `__main__.query_financial_database.invoke` method.
4. Force your mocked instance to return a hardcoded JSON string (e.g., `{"revenue": "10 million"}`).
5. Execute the test using a user prompt asking for financial data. 
6. Assert that the Agent successfully queried the tool, parsed the intercept, and outputted a response containing your mocked values.
