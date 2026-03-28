# Lab: Implementing Long-Term Memory

## The Scenario
Your company is designing a virtual personalized assistant for its top executives. The assistant must remember the executives' names and preferred communication styles even if they close their laptops and return on a completely different device days later (i.e., a different chat thread). You have been tasked with building a LangGraph pipeline that integrates both short-term thread memory (Checkpointer) and long-term global memory (Store) to guarantee a flawless, personalized user experience across sessions.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e050-e050-long_term_memory_lab.py`.
3. Complete the `executive_assistant_node` function:
   - Extract the `"user_id"` from the `config["configurable"]` dictionary.
   - Using the injected `store` object, fetch the `user_profile` from the `"executive_profiles"` namespace using the `user_id`. (Hint: `store.get(...)`).
   - If a profile exists, retrieve the `"name"` and `"style"` from the profile's `.value` dictionary. Construct a personalized greeting using these values appended to the user's input.
   - If a profile does *not* exist, create a new one using `store.put(...)`. Use the namespace `("executive_profiles", user_id)`, the key `"profile"`, and a dictionary value of `{"name": "Guest", "style": "Standard"}`. Construct a generic "first-time" greeting.
4. Complete the `build_assistant_graph` function:
   - Instantiate your `InMemoryStore` and `MemorySaver`.
   - Compile the graph. Crucially, pass BOTH the `checkpointer=` and `store=` arguments to the `compile()` method so the graph engine attaches your memory layers natively.

## Definition of Done
- The script executes successfully.
- Scenario 1 (Initial Thread) correctly greets the user and automatically saves a default profile to the global store because they are new.
- Scenario 2 (New Thread) proves cross-thread persistent memory by correctly identifying the user and applying the formatting styles loaded from the global store, despite having a completely original `thread_id`.
