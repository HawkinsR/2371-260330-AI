# Advanced Agentic Design and MCP

## Learning Objectives

- Manage **Prompt Versioning and Lifecycle** using LangChain Hub and Metadata.
- Implement **Online Evaluation** loops to score agent performance in real-time.
- Design **Human-in-the-Loop (HITL)** approval workflows for sensitive agent actions.
- Explore the **Model Context Protocol (MCP)** for standardized data interaction.
- Integrate **Multi-Agent Orchestration** patterns (Router vs. Supervisor).

## Why This Matters

As agents move from prototypes to production, "set and forget" logic fails. High-stakes applications (like financial transfers or medical advice) require **Human-in-the-Loop** checkpoints to ensure safety. Furthermore, as data sources proliferate, the **Model Context Protocol (MCP)** provides a unified way to connect agents to external systems like GitHub, Slack, or local databases without writing custom API glue-code for every integration.

> **Key Term - Model Context Protocol (MCP):** An open standard that allows AI applications to connect to data sources and tools through a consistent, plug-and-play interface. Instead of rebuilding integrations for every new model or tool, MCP enables a "universal adapter" strategy for agentic data access.

## The Concept

### Prompt Management and Versioning

In production, you should never hardcode long prompts in your Python files.
- **LangChain Hub:** A central repository (like GitHub for prompts) where you can pull specific versions of a prompt.
- **Ab/Versioning:** Testing `v1.2` of a prompt against `v1.3` to see which yields better RAGAS scores.

### Online Evaluation and HITL

Agents can make mistakes. **HITL (Human-in-the-Loop)** ensures that critical steps (like "Send Email" or "Execute Trade") are paused until a human clicks "Approve."
- **Interrupts:** LangGraph allows you to `interrupt` the execution flow, save the state, and wait for external input.
- **Online Evaluation:** Automatically scoring live interactions using a "Judge LLM" to identify failures before the user reports them.

### Multi-Agent Orchestration

Complex tasks are often better handled by a team of agents rather than one "god agent."
- **Router Pattern:** A central LLM looks at the query and decides which specialized agent to call.
- **Supervisor Pattern:** A manager agent delegates tasks to "worker" agents and reviews their work before responding to the user.

> **Key Term - Human-in-the-Loop (HITL):** A design requirement where an AI system requires human intervention at specific decision points to verify output, correct errors, or authorize high-risk actions.

## Code Example

```python
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrock
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# 1. Setup Persistence and Model
memory = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
llm = ChatBedrock(model_id="us.anthropic.claude-3-5-sonnet-20240620-v1:0")

# 2. Compile Agent with HITL Interruption
# We configure the agent to interrupt BEFORE any tool call to allow human review
agent = create_react_agent(
    llm, 
    tools=[some_sensitive_tool], 
    checkpointer=memory,
    interrupt_before=["tools"] 
)

# 3. Execution
config = {"configurable": {"thread_id": "audit-trail-1"}}
query = {"messages": [("user", "Please execute the sensitive financial transfer.")]}

# The agent will stop and save state before calling 'some_sensitive_tool'
agent.invoke(query, config)

# 4. Human Approval (Simulated)
# After the human reviews the 'snapshot', they can 'resume' the thread
# agent.invoke(None, config) 
```

## Additional Resources

- [Model Context Protocol (MCP) Official Site](https://modelcontextprotocol.io/)
- [LangGraph Human-in-the-loop Tutorial](https://langchain-ai.github.io/langgraph/how-tos/human-in-the-loop/)
- [LangChain Prompt Hub](https://smith.langchain.com/hub)
