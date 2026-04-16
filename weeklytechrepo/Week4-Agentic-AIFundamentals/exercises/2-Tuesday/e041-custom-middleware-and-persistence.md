# Exercise 041: Custom Middleware and Persistence

## Overview

In this lab, you will move beyond simple tracing and implement custom middleware for security (PII masking) and a state persistence layer (checkpointer) to allow agents to resume conversations across different sessions.

## Objectives

- Create a middleware function to intercept and mask sensitive user data.
- Implement a mock checkpointer to save and load agent state.
- Simulate a multi-session conversation where the agent remembers past interactions.

## Instructions

1. Open `starter_code/e041-middleware_persistence_lab.py`.
2. **Task 1**: Complete the `pii_masking_middleware` function to redact credit card numbers.
3. **Task 2**: Implement the `save` and `load` methods in the `MockCheckpointer` class.
4. **Task 3**: Run the script and observe how the agent recovers context in 'Session 2'.

## Validation

- Verify that credit cards in the input are replaced with `[REDACTED_CARD]`.
- Confirm that the second query in the script correctly lists the previous interaction from history.
