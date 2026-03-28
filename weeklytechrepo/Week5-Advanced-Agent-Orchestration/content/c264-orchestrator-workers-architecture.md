# Orchestrator-Workers Architecture

## Learning Objectives

- Design hierarchical multi-agent structures utilizing the Orchestrator-Workers Architecture.
- Establish decoupled operational schemas Defining Sub-Agent Interfaces.
- Direct disparate workloads autonomously by Routing Tasks to Sub-Agents.
- Consolidate parallel execution outcomes Aggregating Sub-Agent Outputs.
- Transfer operational control securely implementing Handoffs between Agents.
- Segment token spaces explicitly Managing Sub-Agent Context.
- Instantiate an overarching control layer via Supervisor Implementation.

## Why This Matters

If you ask a single agent to "Write a marketing blog, check it for compliance, format it in HTML, and post it to WordPress," it will likely fail. Single agents get confused when holding too many tools or context domains simultaneously. The enterprise solution is a Multi-Agent system. An **Orchestrator** (or Supervisor) receives the complex goal, breaks it into specialized tasks, delegates those tasks to discrete **Worker** agents (e.g., a Writer Agent, a Compliance Agent, an SEO Agent), and then aggregates the results.

> **Key Term - Multi-Agent System:** An AI architecture where multiple specialized LLM agents work together to complete a complex task, each having its own tools, system prompt, and area of expertise. Rather than one generalist agent trying to do everything, multi-agent systems achieve better results through specialization and parallel execution.

> **Key Term - Orchestrator / Supervisor Agent:** In a multi-agent system, the orchestrator is a specialized node that receives the high-level goal, decides which worker agent should act next, and aggregates results. The orchestrator does not perform work itself — its only "tools" are the names of its worker agents. It functions like a project manager assigning tasks to specialists.

## The Concept

### Defining Sub-Agent Interfaces

Each Worker agent is completely independent. The "Writer Agent" only knows about writing and has access solely to the "Web Search" tool. The "Compliance Agent" only knows legal rules and has access solely to the "Company Policy Retriever" tool.
Crucially, these workers do *not* communicate directly. They communicate strictly by returning their output to the shared global State, which is then parsed by the Orchestrator.

### The Supervisor Implementation

The Orchestrator (Supervisor) is a specialized LLM node. It does *not* possess any standard tools. Instead, its "tools" are the names of its Worker agents.
Its System Prompt is explicitly: "You are a Supervisor. You manage the Writer and Compliance agents. Given a user request, decide which agent should act next, or decide if the task is FINISHED."
The Supervisor's output is purely a routing decision (e.g., "Route to -> Writer Agent").

### Handoffs and Managing Context

When the Orchestrator routes the graph to a Worker, that Worker executes its own internal loop (perhaps it searches the web 5 times). When the Worker is done, it *handoffs* control back to the Supervisor.
To prevent token explosion, the state must be segmented. The Orchestrator only needs to see the *final draft* from the Writer Agent; it does not need to see the 5 failed web searches the Writer Agent performed to get there.

> **Key Term - Handoff (Agent-to-Agent):** The mechanism by which one agent completes its work and returns control to the supervisor/orchestrator. A handoff typically involves updating the shared State with the agent's output and routing the graph back to the Supervisor node. Clean handoffs ensure the Supervisor sees only the relevant summary, preventing its context window from being flooded with the worker's internal deliberations.

## Code Example

```python
from typing import Literal, TypedDict, Annotated
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

# 1. Define the shared State for the multi-agent graph
class BlogState(TypedDict):
    messages: list   # Accumulates the work of all agents
    next_node: str   # Tracks where the Supervisor routes next

# 2. Define the Supervisor's Routing Schema using Pydantic (PascalCase required)
class RouteResponse(BaseModel):
    next_node: Literal["FINISH", "Writer_Agent", "Compliance_Agent"]

llm = init_chat_model("gpt-4o", model_provider="openai", temperature=0)

# 3. Configure the Supervisor Prompt
supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a supervisor managing a conversation between: {members}.
    Given the user request, respond with the worker to act next.
    If the blog is written AND checked for compliance, respond FINISH."""),
    MessagesPlaceholder(variable_name="messages"),
])

# 4. Create the Supervisor Chain (forcing structured output so routing is deterministic)
supervisor_chain = (
    supervisor_prompt
    | llm.with_structured_output(RouteResponse)
)

# 5. Define stub Worker Agent Nodes (each isolated to its own specialization)
def writer_agent_node(state: BlogState):
    """In production: this node runs its own internal ReAct loop with web search tools."""
    print("  [Writer Agent] Drafting the blog post...")
    return {"messages": state["messages"] + ["WRITER: Blog draft complete."]}

def compliance_agent_node(state: BlogState):
    """In production: this node checks the draft against a policy retriever tool."""
    print("  [Compliance Agent] Reviewing draft for policy violations...")
    return {"messages": state["messages"] + ["COMPLIANCE: Draft approved."]}

def supervisor_node(state: BlogState):
    """Routes to the correct next worker or terminates the graph."""
    result = supervisor_chain.invoke({"members": "Writer_Agent, Compliance_Agent", "messages": state["messages"]})
    return {"next_node": result.next_node}

# 6. Assemble the Graph
builder = StateGraph(BlogState)
builder.add_node("Supervisor", supervisor_node)
builder.add_node("Writer_Agent", writer_agent_node)
builder.add_node("Compliance_Agent", compliance_agent_node)

builder.add_edge(START, "Supervisor")
builder.add_conditional_edges("Supervisor", lambda x: x["next_node"],
    {"Writer_Agent": "Writer_Agent", "Compliance_Agent": "Compliance_Agent", "FINISH": END}
)
builder.add_edge("Writer_Agent", "Supervisor")      # Handoff back to Supervisor
builder.add_edge("Compliance_Agent", "Supervisor")  # Handoff back to Supervisor

graph = builder.compile()
print("Orchestrator-Workers graph compiled successfully.")
```

## Additional Resources

- [Multi-Agent Orchestrator Patterns](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [LangGraph Supervisor Tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/)
