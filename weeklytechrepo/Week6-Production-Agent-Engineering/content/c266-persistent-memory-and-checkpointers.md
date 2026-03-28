# Persistent Memory and Checkpointers

## Learning Objectives

- Differentiate between Short-term versus Long-term Memory in agent architectures.
- Instantiate Persistent Checkpointers for Sessions enabling durable multi-turn conversations.
- Intercept and utilize The `Store` Interface for global, state-agnostic memory.
- Segment memory securely establishing strict Namespaces & Scopes.
- Implement explicit memory schemas Storing User Preferences.
- Access memory across independent architectures natively enabling Cross-Thread Memory.
- Contrast retrieval methods comparing RAG vs Parametric vs Ephemeral Memory.

## Why This Matters

Until now, our agents have essentially suffered from amnesia. Whenever a LangGraph execution finishes, the `State` is completely erased from RAM. If the user returns 5 minutes later, the agent has no idea who they are. To build enterprise "assistants" that learn about the user over months, we must integrate persistent checkpointers and global memory stores. This upgrades the agent from a stateless function to a stateful companion.

> **Key Term - Stateless vs. Stateful:** A *stateless* function produces its output purely from its inputs, retaining nothing between calls. A *stateful* system persists data across calls — remembering previous interactions. Most web servers and APIs are stateless for scalability; AI assistants must be stateful to feel like genuine collaborators rather than amnesiacs.

## The Concept

### Ephemeral Memory vs RAG vs Persistent Components

- **Parametric Memory:** Information baked into the LLM's raw weights during training (e.g., knowing that Paris is in France).
- **Ephemeral Memory:** The `messages` list within a single LangGraph State execution. It is deleted the moment the graph finishes running.
- **RAG (Long-term):** Searching a Vector Database for external facts.
- **Persistent Semantics (Long-term):** Saving user-specific facts (e.g., "The user likes dark mode") to a rigid database using LangGraph's Checkpointer and Store.

> **Key Term - Parametric Memory:** Knowledge encoded directly into an LLM's neural network weights during the training process. This is the model's "built-in" knowledge — it cannot be updated without retraining the model. Example: GPT-4 knows the rules of chess without being told in the prompt.

> **Key Term - Ephemeral Memory:** Data that exists only for the duration of a single execution and is discarded immediately afterward. In LangGraph, the `State` TypedDict is ephemeral — it lives in RAM for one graph run and is gone forever once the run completes.

### Persistent Checkpointers and The Store

- **Checkpointers** save the exact `State` of a specific thread (conversation). If you pass the same `thread_id` tomorrow, you resume the exact same conversation where it left off.
- **The Store** acts as global memory *across* threads. A checkpointer cannot easily share information between "Thread 1" and "Thread 2". By passing a `Store` (like Postgres) to the graph, the agent can write preferences (e.g., {"user_id": 123, "tone": "formal"}) into a distinct **Namespace**. Regardless of which thread is active, if the `user_id` matches, the agent reads the Store to dictate its behavior.

> **Key Term - The Store (LangGraph):** A key-value database layer attached to a compiled graph that persists data *across* different conversation threads. While a Checkpointer remembers the history of one specific conversation, the Store remembers facts about specific users or contexts regardless of which conversation they are in. Think of it as the agent's long-term personal memory.

> **Key Term - Namespace (Memory Scoping):** A hierarchical key used to organize entries in the Store, typically structured as a tuple like `("user_preferences", user_id)`. Namespaces prevent one user's data from colliding with another's, and allow different categories of memory (preferences vs. facts vs. history) to be stored and retrieved independently.

## Quick Reference: Choosing the Right Memory Type

| Memory Type | Lifespan | Best Used For |
|---|---|---|
| Ephemeral (`State`) | Current graph run only | Messages within one conversation turn |
| Checkpointer | Across sessions (same `thread_id`) | Resuming a multi-day conversation |
| Store | Permanently, across all threads | User preferences, profile facts |
| RAG (Vector DB) | Permanently in external DB | Company knowledge base, documentation |
| Parametric | Baked into model weights | General world knowledge (no code needed) |

## Code Example

```python
from langgraph.store.memory import InMemoryStore # Use Postgres in Production
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

# 1. Initialize Long-Term Global Store & Short-Term Thread Checkpointer
store = InMemoryStore()
checkpointer = MemorySaver()

# 2. Example Node reading from the Global Store
def greeting_node(state: dict, config: dict, store: InMemoryStore):
    # Extract the user ID passed in at invocation
    user_id = config["configurable"].get("user_id", "anonymous")
    
    # Check the Global Store for cross-thread memory
    namespace = ("user_preferences", user_id)
    user_data = store.get(namespace, "profile")
    
    # If we remember them, use their preference!
    if user_data:
        tone = user_data.value.get("preferred_tone", "friendly")
        return {"response": f"[Tone: {tone}] Welcome back, user {user_id}!"}
    
    return {"response": "Hello, I don't believe we've met."}

# 3. Compile Graph with Persistent Memory
builder = StateGraph(dict)
builder.add_node("Greeting", greeting_node)
builder.add_edge(START, "Greeting")
builder.add_edge("Greeting", END)

# Attach both the checkpointer AND the store
graph = builder.compile(checkpointer=checkpointer, store=store)

# 4. Global Memory Injection
# We manually write a preference to the Store.
# (In reality, an LLM Tool would write this autonomously after asking the user!)
store.put(
    ("user_preferences", "user_123"),
    "profile",
    {"preferred_tone": "Formal and concise"}
)

# 5. Invocation
config = {"configurable": {"thread_id": "thread_A", "user_id": "user_123"}}
result = graph.invoke({"input": "Hi"}, config=config)
print(result["response"])
```

## Additional Resources

- [LangGraph Persistence Concepts](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [LangGraph Store Documentation](https://langchain-ai.github.io/langgraph/concepts/store/)
