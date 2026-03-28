# Self-Correction and Plan and Execute

## Learning Objectives

- Architect robust agent workflows utilizing Self-Correction Loops.
- Separate generation from verification utilizing Plan-and-Execute Agents.
- Force internal evaluation inserting explicit Critique Nodes & Prompts.
- Engineer state transitions driving Iterative Refinement Patterns.
- Mitigate data pollution configuring strict Output/Input Validation Guards.
- Catch errors before the user sees them Handling Hallucinations with Reflection.

## Why This Matters

Asking an LLM to generate complex code, math, or legal structures in a single "zero-shot" response is a recipe for hallucinations. Human engineers don't write perfect software on the first try; we write, test, debug, and rewrite. Agentic flows must adopt this **Reflection** pattern. By separating the "Generator" agent from the "Critique/Evaluator" agent, we can build loops that refuse to output a final answer until the Output Validation Guards are fully satisfied.

> **Key Term - Zero-Shot:** Asking an LLM to complete a task with no examples or prior reasoning steps — just a raw prompt and an expected output. Zero-shot works for simple requests but fails for complex, multi-step, or format-sensitive tasks because the model has no scaffolding to guide its reasoning.

> **Key Term - Reflection (AI Pattern):** An architectural pattern where an AI's output is fed back into the system for self-critique before being delivered to the user. The model (or a separate evaluator model) reviews its own work, identifies flaws, and iterates until quality standards are met — mirroring how a human engineer reviews their own code before committing it.

## The Concept

### Plan-and-Execute Agents

Instead of jumping straight into action, a Plan-and-Execute architecture behaves like a senior engineer defining a roadmap for a junior developer.

1. **Planner Node:** Takes the user query and generates a step-by-step checklist (using structured JSON output).
2. **Executor Node:** Takes Step 1 on the list, uses tools to accomplish it, and saves the observation.
3. **Controller Node:** Checks if Step 1 was actually successful. If yes, it moves to Step 2. If no, it updates the plan and sends it back to the Executor.

> **Key Term - Plan-and-Execute Architecture:** An agent design pattern that separates *planning* from *acting*. A dedicated Planner LLM first generates a structured list of steps needed to accomplish the goal, then a separate Executor agent works through those steps one-by-one. This prevents "action before thinking" failures and makes complex multi-step tasks manageable and debuggable.

### Self-Correction and Critique Nodes

The Reflection pattern forces the graph to critique its own work.

- Node A generates a draft response.
- Node B (Critique) reads the draft *and* the original prompt. It is instructed explicitly: "Find all errors, hallucinations, and missing requirements in this draft."
- If Node B finds an error, the graph routes *backward* to Node A, passing Node B's critique as feedback. Node A tries again.
- The graph only routes to the `END` node when Node B returns a perfect score (or a max loop limit is hit).

Choosing the right revision limit is a deliberate trade-off: a higher limit (e.g., 5) produces higher quality outputs but multiplies API calls and cost linearly. For most production systems, a limit of 2–4 iterations strikes the right balance — lower for fast/cheap use cases like customer service replies, higher for critical documents like legal summaries or financial disclosures.

### Resolving Validation Errors

When forcing an LLM to output a Pydantic schema, it sometimes hallucinates formatting (like responding with plain text instead of JSON). Instead of crashing the app, LangGraph catches the `ValidationError`. The error string itself is passed *back* into the LLM prompt as context ("You failed to format as JSON, here is the error: ... Try again."), creating an automatic retry loop.

> **Key Term - ValidationError (Pydantic):** An exception raised when data received from an LLM does not conform to the expected Pydantic schema — for example, a required field is missing, a number is passed where a string is expected, or the entire response is plain text instead of JSON. In self-correcting loops, the ValidationError message itself becomes the feedback fed back to the LLM to guide its retry.

## Code Example

```python
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END

# 1. State holding the Draft and the Critique
class ReflectionState(TypedDict):
    task: str
    draft: str
    critique: str
    revision_count: int

# 2. Generator Node (Writes the initial draft OR rewrites based on critique)
def generate_draft(state: ReflectionState):
    if state.get("critique"):
        print(f"\n[Generator] Applying Feedback: {state['critique']}")
        return {"draft": f"{state['draft']} + (Revised with logic)", "revision_count": state["revision_count"] + 1}
    
    print("\n[Generator] Writing Initial Draft...")
    return {"draft": "The sky is green.", "revision_count": 0}

# 3. Critique Node (Acts as the harsh reviewer)
def critique_draft(state: ReflectionState):
    print(f"\n[Evaluator] Reviewing Draft: '{state['draft']}'")
    
    # Fake LLM logic: We know the sky isn't green.
    if "green" in state["draft"].lower():
        return {"critique": "Hallucination detected. The sky is blue. Please correct."}
    
    return {"critique": "PERFECT"}

# 4. Conditional Routing (The Loop!)
def routing_logic(state: ReflectionState):
    if state["critique"] == "PERFECT":
        return END # Release to the user
    
    if state["revision_count"] >= 3:
        return END # Prevent infinite loops!
        
    return "Generator" # Send it back to the generator

# 5. Build the Self-Correction Graph
builder = StateGraph(ReflectionState)
builder.add_node("Generator", generate_draft)
builder.add_node("Evaluator", critique_draft)

builder.add_edge(START, "Generator")
builder.add_edge("Generator", "Evaluator") # Always evaluate after generating
builder.add_conditional_edges("Evaluator", routing_logic) # Route based on critique

graph = builder.compile()

print("--- Starting Execution ---")
graph.invoke({"task": "What color is the sky?", "revision_count": 0})
```

## Additional Resources

- [Plan-and-Execute Agents in LangGraph](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/)
- [Reflection Agent Tutorial](https://langchain-ai.github.io/langgraph/tutorials/reflection/reflection/)
