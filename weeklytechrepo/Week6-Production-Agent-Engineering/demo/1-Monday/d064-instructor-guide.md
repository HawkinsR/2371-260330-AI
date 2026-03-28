# Demo: Persistent Memory and Checkpointers

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
| --- | --- |
| **Stateless vs. Stateful** | *"A basic Python function like `def add(a, b): return a + b` is stateless. What would make it stateful? And why does making an AI assistant stateful create engineering complexity that doesn't exist in a regular function?"* |
| **Memory Types** | *"List the three types of memory an AI agent can use. If a user tells the bot 'I prefer formal responses' on Monday, which memory type needs to store that so the bot remembers it on Friday?"* |
| **The Store vs. Checkpointer** | *"A Checkpointer remembers one conversation. A Store remembers across all conversations. If two different users both interact with the same agent, can their Store memories collide? How does a Namespace prevent that?"* |
| **Cross-Thread Memory** | *"Why would saved preferences in Thread A not automatically be visible in Thread B without a Store? What Python data structure does a Namespace most resemble, and why is that mental model helpful?"* |


## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/persistent-memory-flow.mermaid`.
2. Trace the path from the User Request. Explain the critical difference between the two memory types:
   - **Multi-Turn Memory (Checkpointer):** Remembers the exact back-and-forth of the current conversation (e.g., "What was the last thing I said?"). Tied to `thread_id`.
   - **Persistent State (The Store):** Remembers global facts about the user regardless of the conversation (e.g., "Always speak to me in Spanish"). Tied to `user_id`.
3. **Discussion:** Ask the class: "If a user explicitly asks the agent to remember their favorite color, does the agent use RAG, the Checkpointer, or the Store to fulfill this request?" (Answer: The Store. RAG is for external documents; Checkpointers are ephemeral to the session. The Store is the dedicated 'Brain' for user preferences).

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d064-cross-thread-memory-store.py`.
2. Walk through the `greeting_and_memory_node` function.
   - Show how the `store: InMemoryStore` is injected directly into the function signature.
   - Highlight the Namespace logic: `("user_preferences", user_id)`. This prevents Alice from accidentally reading Bob's preferences!
3. Review the `build_graph()` function to explicitly show the `checkpointer` and `store` being passed into `.compile()`.
4. Execute the script via `demonstrate_memory_systems()`. 
5. The terminal output is split into 3 Scenarios. Walk the students through Scenario 3 carefully:
   - "Look at Thread B. The `messages` array only has 1 message in it. The Checkpointer memory is blank because it's a new thread. BUT, the agent still responded with a `[FORMAL TONE]`. Why? Because the `user_id` allowed it to query the Global Store across threads."

## Summary
Reiterate that separating conversational state from persistent user state is the foundational architecture required to build intelligent, personalized AI products rather than simple stateless chatbots.
