# c262: LangGraph Fundamentals

## Introduction to LangGraph & Agents as Graphs Overview
LangGraph is a library for building stateful, multi-actor applications with LLMs. Unlike standard linear chains, LangGraph models Agents as Graphs. This approach allows for cycles, which are essential for iterative processes like reflection and error correction.

## LangChain vs LangGraph
- **LangChain**: Focuses on constructing Directed Acyclic Graphs (DAGs) and linear chains.
- **LangGraph**: Designed for cyclic graphs, where agents can loop iteratively, maintain state across multiple turns, and handle asynchronous interruptions.

## Core Components: Nodes, Edges, Graphs, Graph Errors
- **Nodes**: Python functions or LLM calls that process the state.
- **Edges & Conditional Edges**: Normal edges provide direct links between nodes. Conditional Edges use a router function to decide the next node based on the state.
- **Graphs**: The overarching structure connecting nodes and edges.
- **Graph Errors**: Handled by adding fallback nodes or conditional edges that trigger upon validation failures.

## StateGraph vs create_agent
- **`create_agent`**: A legacy wrapper for quickly spinning up simple agents.
- **`StateGraph`**: The modern, core foundation of LangGraph applications. It defines exactly how the workflow operates by managing the graph directly.

## TypedDict State
State in LangGraph is defined using a Python `TypedDict`. This shared data structure allows you to declare exact fields and use reducers (like `Annotated[list[str], add]`) so that new messages or updates are appended rather than overwritten.

## Memory and Persistence
LangGraph uses Checkpointers (e.g., `MemorySaver`) for persistence. This memory allows threads to be paused, resumed, and maintain state over long user sessions.

## Binding the Tools & Streaming
- **Binding the Tools**: LLM nodes must have tools explicitly bound via `.bind_tools(tools)` so they know what functions they can call.
- **Streaming**: LangGraph natively supports streaming state updates (e.g., streaming node completions or token-by-token outputs to the user interface).

## Command for State Updates
The `Command` API allows nodes to directly return state updates and precise routing instructions in one dictionary structure:
```python
return Command(update={"steps": state["steps"]+1}, goto="next_node")
```
