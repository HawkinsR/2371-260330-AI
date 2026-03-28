# Runtime Configuration and Middleware

## Learning Objectives

- Abstract environment variables using generalized Runtime Configuration (`configurable`).
- Expose global memory directly to isolated functions Passing State to Tools (`ToolRuntime`).
- Intercept and format input/output pathways via Middleware Patterns (`@wrap_model_call`).
- Architect persistent memory pipelines Managing Conversation History.
- Mitigate token limits structurally by Trimming Messages for Context.
- Distribute and sync state variables ensuring Shared State across Nodes.

## Why This Matters

Hardcoding variables into your AI graph (like setting `model_name = "gpt-4"`) means you have to rewrite code every time you switch environments or test new models. Enterprise architectures dynamically inject Runtime Configurations when the graph is invoked. Furthermore, as graphs run longer loops, the `messages` list in the State grows exponentially. If you do not actively prune and trim the conversation history middleware, the LLM will inevitably crash due to Context Window overflow limits.

> **Key Term - Runtime Configuration:** Values that are passed into a graph at invocation time rather than hardcoded during development. This allows a single compiled graph to behave differently depending on who is calling it (e.g., different user roles, different model providers, different feature flags) without rebuilding or redeploying the graph code.

> **Key Term - Middleware (in AI):** A processing layer that sits between two components and intercepts/transforms data passing through. In LangGraph, middleware can intercept the message list before it reaches the LLM node, performing operations like trimming old messages, injecting system context, or logging — without those operations being part of the core node logic.

## The Concept

### Runtime Configuration

When you compile a LangGraph, you can pass a `configurable` dictionary at invocation time. This allows a single deployed graph to act fundamentally differently depending on who called it. You could pass `{"configurable": {"user_id": "123", "strict_mode": True}}`. Nodes can read this configuration via the `RunnableConfig` object without it needing to pollute the core `State` dictionary.

### ToolRuntime and State

Sometimes, a `@tool` needs to know something about the state that the LLM shouldn't necessarily have to pass to it explicitly. For example, a `delete_account` tool needs the `user_id`. Instead of forcing the LLM to pass the ID (which it might hallucinate), you can use `ToolRuntime` artifacts to inject the `user_id` from the Graph's State directly into the tool's execution context securely.

### Managing and Trimming Conversation History

LLMs charge by the token. Leaving 50 turns of "Hello" and "How are you?" in the message history is a massive waste of money and compute. We utilize LangChain's `trim_messages` utility as middleware. Before the State's message list is passed to the LLM node, the trimmer calculates the token count. If it exceeds a boundary (e.g., 4000 tokens), it drops the oldest messages, keeping only the System Prompt and the most recent, relevant context.

> **Key Term - Message Trimming:** The technique of pruning a conversation history before passing it to an LLM, removing older messages to stay within token and cost limits. A well-configured trimmer always preserves the System Prompt (which establishes the AI's rules) and the most recent messages (which provide immediate context), discarding older exchanges that are no longer relevant to the current question.

## Code Example

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model

# --- Runtime Configuration Pattern ---
# When invoking the graph, pass a configurable dict:
# graph.invoke(state, config={"configurable": {"user_id": "abc123", "model": "gpt-4o"}})
# Inside any node function, read it via the RunnableConfig argument:
#
# def my_node(state: AgentState, config: RunnableConfig):
#     user_id = config["configurable"].get("user_id")
#     model_name = config["configurable"].get("model", "gpt-3.5-turbo")
#     # This lets one deployed graph serve many different users safely.

# 1. Create a dummy message history that is far too long
messages = [
    SystemMessage("You are a helpful assistant. Never forget this."),
    HumanMessage("Hi, I like apples."),
    AIMessage("Great! Apples are tasty."),
    HumanMessage("I also like cars."),
    AIMessage("Cars are fast."),
    HumanMessage("What is my favorite fruit?"), # The most recent, relevant turn
]

print(f"Original Message Count: {len(messages)}")

# 2. Configure the Trimmer (Acting as Middleware)
# We want to keep the final conversation turns, but NEVER drop the SystemPrompt at index 0
trimmer = trim_messages(
    max_tokens=45,         # Extremely low for demonstration purposes
    strategy="last",       # Keep the last N tokens worth of messages
    token_counter=init_chat_model("gpt-3.5-turbo"), # Pass the model instance directly
    include_system=True,   # Critical: Always keep the system instructions
    allow_partial=False    # Do not cut a message string in half
)

# 3. Execute the Trimmer
trimmed_messages = trimmer.invoke(messages)

print(f"Trimmed Message Count: {len(trimmed_messages)}")
print("\nRetained Messages:")
for msg in trimmed_messages:
    print(f"- [{type(msg).__name__}]: {msg.content}")

# In a LangGraph, you would call `trimmer.invoke(state["messages"])` 
# right before passing the state to the LLM node!
```

## Additional Resources

- [LangChain Message Trimming](https://python.langchain.com/docs/how_to/trim_messages/)
- [LangGraph Configuration Tutorial](https://langchain-ai.github.io/langgraph/how-tos/configuration/)
