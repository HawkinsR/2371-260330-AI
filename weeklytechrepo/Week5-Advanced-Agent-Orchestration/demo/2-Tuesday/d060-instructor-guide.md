# Demo: LangGraph State and Routing

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
| --- | --- |
| **LangGraph vs. Simple Chain** | *"With a simple LangChain chain (A → B → C), what happens when the AI's B step fails and we need to retry just B, not A? How does a StateGraph help here?"* |
| **Graph State (TypedDict)** | *"In a StateGraph, two nodes run back-to-back. How does Node B know what Node A produced? Does Node B import Node A and call it? Or is there a different mechanism?"* |
| **Conditional Edges** | *"A customer service bot must route 'billing' questions to the Billing Node and 'technical' questions to the Tech Node. Write out in plain English (not code) the logic the conditional edge function should implement."* |
| **Compiling the Graph** | *"`builder.compile()` locks the graph's structure. What's the benefit of having an immutable compiled graph vs. a graph that can be modified at runtime?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/stategraph-compilation.mermaid`.
2. Trace the path from the `START` node. Call attention to the `Global Memory` TypedDict hovering above the graph.
3. **Discussion:** Ask the class: "When Node A (Categorizer) is running, can it read the `final_answer` variable in the `TypedDict`?" (Answer: Yes, but the `final_answer` will be empty or null because Node 2/3 hasn't written to it yet. State is global and mutable; execution order dictates what data is available when).
4. Explain the Conditional Router. Contrast this with `graph.add_edge()`. An `edge` is a static, unbreakable pipeline. A `conditional_edge` allows mathematical "If/Else" logic to dynamically define the architecture at runtime.

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d060-stategraph-compilation.py`.
2. Highlight `class AgentState(TypedDict)`: This explicit class defines exactly what keys the graph tracks. No dictionary key errors allowed.
3. Review `categorize_node`. Note that it only returns `{"category": cat, "sentiment": sent}`. It does NOT return the whole state. LangGraph implicitly takes this dictionary and merges it into the Global State automatically.
4. Walk through `route_by_category`. Notice how it returns a string like `"manager_escalation"`.
5. Review `build_custom_graph`. Point out the mapping dictionary inside `add_conditional_edges`. The string returned by the router must map perfectly to an added Node name or `END`.
6. Execute the script via `run_scenarios()`. 
7. Follow the terminal logs for Scenario C. Explain that even though the Categorizer identified it as a "billing" issue, the `route_by_category` function had an explicit rule that overrode standard routing based on the "urgent" sentiment, bypassing the billing node entirely.

## Summary
Reiterate that mastering StateGraphs allows AI engineers to build deterministic safety rails (using normal Python code) around non-deterministic LLM behavior. You cannot control what an LLM says perfectly, but you can control exactly *when* and *where* those LLM nodes execute.
