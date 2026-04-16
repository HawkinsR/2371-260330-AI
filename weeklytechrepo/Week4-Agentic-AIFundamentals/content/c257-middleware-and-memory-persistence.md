# Middleware and Memory Persistence

## Learning Objectives

- Implement Prebuilt and Custom Middleware to intercept and secure agent messages.
- Configure **LangGraph Persistence** to save and resume agent state across sessions.
- Apply **Injection & Stuffing Guardrails** to protect against malicious prompts.
- Understand the strategy for **Multi-Agent Context Handoffs** (State Sharing).
- Establish State Persistence using Checkpointers and Step-wise Scratchpads.

## Why This Matters

As agent conversations grow longer, the token window becomes a critical bottleneck. **Memory Persistence** and **Middleware** provide the architectural solves. Middleware allows you to intercept and modify messages before they reach the model (e.g., for PII masking or injection defense), while persistent memory ensures an agent "remembers" a user across multiple sessions. In production, this means using a database-backed checkpointer so that an agentic workflow can survive server restarts and user disconnects.

> **Key Term - State Persistence:** The ability to save the current state of an agent (its variables, message history, and tool outputs) to a database. Using a **Checkpointer** allows an agent to resume a conversation exactly where it left off, even days later, by referencing a unique `thread_id`.

## The Concept

### Middleware and Security Guardrails

In LangChain, **Middleware** (implemented as custom nodes or functional wrappers) allows you to process data at the "edges" of your logic.
- **Injection Defense:** Middleware can scan user input for "Prompt Injection" (e.g., "Ignore your previous instructions").
- **PII Masking:** Automatically redacting emails or credit card numbers before they are sent to the LLM.
- **Guardrails:** Rules that "fence in" the LLM's output or input to ensure safety and compliance.

### Memory: Trimming and Persistence

Pure LLMs are stateless. To provide a "chat" experience, we must send the history back with every request.
- **Short-Term Memory:** Stored in the immediate context window during a single session.
- **Long-Term Memory:** Stored in an external database (Checkpoint) like SQLite or Postgres.
- **Trimming / Sliding Window:** An optimization where only the `N` most recent messages are kept to save tokens and prevent context overflow.

### Multi-Agent Context Handoffs

In advanced architectures, a single agent may not be enough. **Multi-Agent Handoffs** occur when one agent (e.g., a "Router") passes the conversation state to another specialized agent (e.g., a "Billing Expert"). The state must be preserved during this handoff so the second agent doesn't lose the user's initial context.

> **Key Term - Multi-Agent Handoff:** A design pattern where an agent "delegates" a task to another agent. This requires a shared state object so the new agent knows what has already been discussed and what tools have been triggered.

## Code Example

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

# 1. Setup Persistence Layer (SQLite)
# In production, this would be a Postgres or Cloud DB
conn = sqlite3.connect("agent_memory.db", check_same_thread=False)
memory = SqliteSaver(conn)

# 2. Initialize Model and Agent
llm = init_chat_model(model="us.anthropic.claude-3-5-sonnet-20240620-v1:0", model_provider="bedrock")
agent = create_react_agent(llm, tools=[], checkpointer=memory)

# 3. Execution with a Thread ID
# The 'thread_id' is how the agent looks up its memory in the database
config = {"configurable": {"thread_id": "user-session-123"}}

# First interaction
input_1 = {"messages": [("user", "My name is Richard and I love Python.")]}
agent.invoke(input_1, config)

# Second interaction (The agent 'remembers' the name from the DB)
input_2 = {"messages": [("user", "What is my name and what is my favorite language?")]}
response = agent.invoke(input_2, config)

print(response["messages"][-1].content)
# Output: "Your name is Richard and you mentioned you love Python."
```

## Additional Resources

- [LangGraph Persistence (Checkpointers)](https://langchain-ai.github.io/langgraph/how-tos/persistence/)
- [LangChain Security Guardrails](https://python.langchain.com/docs/guides/security/)
- [Handling Multi-Agent State](https://langchain-ai.github.io/langgraph/how-tos/multi-agent-handoff/)
