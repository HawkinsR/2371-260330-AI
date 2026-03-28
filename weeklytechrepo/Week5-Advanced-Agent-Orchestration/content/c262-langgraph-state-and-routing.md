# LangGraph State and Routing

## Learning Objectives

- Contrast the rigid `create_agent` wrapper versus bespoke `StateGraph` compilation.
- Model the cognitive memory of an agent by Defining Graph State with `TypedDict`.
- Architect workflows visually configuring Nodes & Edges Basics.
- Inject decision-making logic utilizing Conditional Edges & Routing.
- Finalize and parse the execution matrix by Compiling the Graph.
- Rapidly iterate architectures utilizing LangSmith Studio Prototyping.
- Implement robust fault tolerance methodologies Handling Graph Errors natively.

## Why This Matters

In Week 4, we used `create_react_agent`. That is a pre-built, black-box loop. It is great for simple Q&A, but insufficient for enterprise workflows. What if we want the agent to draft an email, but *force* it to pass the draft to a strict "Compliance Node" before sending? We cannot do that with a black-box agent. **LangGraph** allows us to define the precise State of our application and draw explicit Nodes (Python functions) and Edges (directional arrows) to create totally custom, cyclical, fault-tolerant AI architectures.

> **Key Term - LangGraph:** A framework built on top of LangChain that allows developers to define AI agent workflows as explicit directed graphs. Unlike simple chains (A → B → C), LangGraph supports cycles (A → B → A), parallel branches, and stateful memory, making it suitable for complex, multi-step, enterprise-grade AI workflows.

> **Key Term - StateGraph:** The core LangGraph class. A `StateGraph` compiles a Python `TypedDict` schema (the State) and a set of Nodes and Edges into an executable, stateful AI workflow. Think of it as a flowchart where each box is a Python function and arrows are the routing logic.

## The Concept

### Graph State (`TypedDict`)

The "State" is the global memory of the graph. It is a Python `TypedDict` that is passed from Node to Node. When a Node finishes executing, it returns a dictionary. LangGraph takes that returned dictionary and *updates* the global State. Every node has access to the updated state when it begins executing.

> **Key Term - Graph State (`TypedDict`):** A Python typed dictionary that acts as the shared memory of a LangGraph workflow. Every node function reads from and writes to this shared State object. This allows information to flow between completely isolated nodes without requiring direct function calls between them — nodes communicate through the State, not each other.

### Nodes and Edges

- **Nodes:** Standard Python functions. They take the current `State` as input, do some work (like calling an LLM or a Tool), and return a state update.
- **Edges:** The static pathways connecting nodes. (e.g., `graph.add_edge("node_A", "node_B")` means B always happens after A).

### Conditional Edges

This is where the magic happens. A conditional edge is a Python function that looks at the current State and mathematically decides where the graph should go next. For example, a `router_function` might look at the State's `sentiment` key. If `sentiment == "angry"`, it routes the graph to the `human_escalation_node`. Otherwise, it routes to the `auto_reply_node`.

> **Key Term - Conditional Edge:** A routing function in LangGraph that examines the current State and returns the name of the next node to execute. Unlike static edges (which always go A → B), conditional edges enable dynamic branching — the workflow's path through the graph depends on the data in the State, enabling complex if/else and multi-path logic.

## Code Example

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. Define the Global State Memory
class AgentState(TypedDict):
    user_input: str
    category: str
    final_answer: str

# 2. Define the Nodes (Python functions that update the State)
def categorize_input(state: AgentState):
    """Fakes an LLM categorizing the input."""
    input_text = state["user_input"].lower()
    cat = "billing" if "price" in input_text else "technical"
    return {"category": cat} # Returns a dict to update the global state

def billing_node(state: AgentState):
    return {"final_answer": "Please check your invoice on the billing page."}

def technical_node(state: AgentState):
    return {"final_answer": "Have you tried turning it off and on again?"}

# 3. Define the Conditional Routing Logic
def route_by_category(state: AgentState):
    """Reads the state and returns the string name of the next node."""
    if state["category"] == "billing":
        return "BillingTeam"
    return "TechSupport"

# 4. Build the Graph
builder = StateGraph(AgentState)

builder.add_node("Categorizer", categorize_input)
builder.add_node("BillingTeam", billing_node)
builder.add_node("TechSupport", technical_node)

# Flow: Start -> Categorizer -> Route (Conditional) -> Billing OR Tech -> End
builder.add_edge(START, "Categorizer")
builder.add_conditional_edges("Categorizer", route_by_category)
# Always add explicit edges to END for every terminal node.
# This is especially important in checkpointed graphs (with MemorySaver),
# where ambiguous termination can cause unexpected suspension.
builder.add_edge("BillingTeam", END)
builder.add_edge("TechSupport", END)

# 5. Compile and Invoke
graph = builder.compile()
result = graph.invoke({"user_input": "What is the price of this software?"})
print(f"Determined Category: {result['category']}")
print(f"Final Output: {result['final_answer']}")
```

## Additional Resources

- [LangGraph Conceptual Guide](https://langchain-ai.github.io/langgraph/concepts/)
- [StateGraph API Reference](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.StateGraph)
