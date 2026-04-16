# Exercise e044: Production-Grade Agent Security

## Overview
In this capstone exercise, you will build a **Multi-Agent Supervisor** system with a **Human-in-the-Loop (HITL)** approval gate. A supervisor agent will delegate tasks to specialized worker agents. Before any final action is taken, execution will pause for human approval using LangGraph's `interrupt_before` mechanism.

## Learning Outcomes
- Implement the **Supervisor pattern** for multi-agent delegation.
- Use `interrupt_before` to create a mandatory human approval checkpoint.
- Resume a paused graph from its persisted `thread_id` by invoking with `None`.
- Understand how state integrity is maintained across the pause/resume cycle.

## Prerequisites
- AWS credentials set in environment with Bedrock access.
- `pip install langchain-aws langgraph`

## Instructions

Open `starter_code/e044-multi-agent-hitl-lab.py` and complete the TODOs:

1. **Define Worker Tools** — Write at minimum two `@tool` functions: `search_policy(topic: str)` and `get_employee_data(emp_id: str)`.
2. **Create Worker Agents** — Use `create_react_agent` to build a `policy_agent` and an `hr_agent`, each with their relevant tool only.
3. **Build the Supervisor Graph** — Add a `router_node` that uses the message content to decide which worker to call. Add a `verify_node` that prints an approval notice and pauses.
4. **Compile with Persistence + Interrupt** — Use `SqliteSaver` and `interrupt_before=["verify_node"]`.
5. **Run Session 1** — Invoke with an initial query. Confirm the graph halts at the interrupt.
6. **Run Session 2** — Pass `None` with the same `thread_id` to resume and receive the final answer.

## Deliverable
Demonstrate the full pause → human-checks-state → resume → final-answer lifecycle in your terminal output.
