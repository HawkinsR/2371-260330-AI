# Demo: Orchestrator-Workers Architecture

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
| --- | --- |
| **Why Single Agents Fail** | *"Give me a concrete example where stuffing too many responsibilities into a single agent's system prompt would cause confusing or contradictory behavior. What's the symptom you'd see?"* |
| **Orchestrator / Supervisor Logic** | *"The Supervisor's only 'tools' are the names of its workers. It doesn't write, check compliance, or do SEO. What DOES it do? What does it output on each loop?"* |
| **Worker Specialization** | *"The Writer Agent only has access to a Web Search tool. The Compliance Agent only has access to the Policy Retriever. Why is this restriction important? What breaks if workers have unrestricted tool access?"* |
| **Handoff Back to Supervisor** | *"Why does the Writer Agent return its draft to the Supervisor rather than routing directly to the Compliance Agent? What extensibility benefit does this hub-and-spoke model provide?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/orchestrator-workers.mermaid`.
2. Trace the **hub-and-spoke** model. The Supervisor sits in the center. The workers sit on the edges.
3. **Discussion:** Ask the class: "Why does the Writer Agent not simply pass its draft directly to the Compliance Agent? Why does it handoff back to the Supervisor first?" (Answer: Tight coupling creates brittle systems. If the Writer dictates the flow, what happens when we want to add an SEO agent later? By returning control to a central Supervisor, the Orchestrator acts as the sole brain managing the order of operations, making the architecture highly extensible).

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d062-supervisor-agent-implementation.py`.
2. Review the `MultiAgentState`. Notice how each worker "owns" a specific piece of the state (Writer owns `draft`, Compliance owns `compliance_check`, SEO owns `seo_keywords`).
3. Review `supervisor_node()`. Explain that in LangGraph, this logic is usually handled by an LLM strictly prompted to return a node name using `.with_structured_output()`, but here we use a Python `if/elif` chain to visualize the identical decision tree deterministically.
4. Execute the script via `run_orchestrator()`.
5. Point out the terminal logs. Watch the Supervisor continually re-evaluate the state at the start of every loop. It sees a completed draft, triggers Compliance. It sees a passed Compliance, triggers SEO. It sees SEO output, and finally triggers FINISH. 

## Summary
Summarize that Single Agents fail at complex tasks because their system prompts become schizophrenic ("Be a creative writer, but also a strict lawyer, and also an SEO expert"). Multi-Agent architecture solves this by allowing each node to have a hyper-focused system prompt, governed by an executive Orchestrator.
