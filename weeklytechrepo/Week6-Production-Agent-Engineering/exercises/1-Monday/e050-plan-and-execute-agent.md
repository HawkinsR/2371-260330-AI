# e050: Plan and Execute Architecture

## Objective
Build an asynchronous Plan-and-Execute agent that utilizes the LangGraph `InMemoryStore` to fetch namespace user preferences. By the end of this lab, your agent should output dynamic solutions uniquely formatted to a specific localized user.

## Instructions
1. Open `starter_code/e050-plan-and-execute.py`.
2. Observe the asynchronous `planner_node` architecture and the pre-loaded `executor_node`.
3. Locate the `TODO` block to implement the `InMemoryStore`. Save a `"verbosity"` variable under the namespace list `("user_preferences", "user_123")` with a value of `"extreme"`.
4. In the planner, extract the verbosity preference from the `store` argument. Dynamically inject that preference into your `ChatBedrock` prompt so it tailors the instructions.
5. Compile the script utilizing both your `MemorySaver` checkpointer and your newly configured global `InMemoryStore`.
6. Trigger the `.astream()` invocation to ensure your execution stream proceeds asynchronously and answers using extreme verbosity.
