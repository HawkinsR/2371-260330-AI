# Instructor Guide: Advanced Agentic Design and MCP

## Overview
This demo demonstrates two capstone-level patterns: **Multi-Agent Orchestration** using a Supervisor pattern and **Human-in-the-Loop (HITL)** checkpoints that pause execution to wait for human approval before sensitive actions are taken.

## Phase 1: The Concept (Whiteboard)
**Time:** 10 mins

1.  **Open `diagrams/multi_agent_hitl.mermaid`**.
2.  **Why Multiple Agents?**: Explain the scalability argument — one "god agent" with 20 tools becomes slow and inaccurate. Specializing agents (Research, Finance, HR) produces better results via focused prompts and smaller tool sets.
3.  **The Supervisor Pattern**: Show how the Supervisor acts as a "manager." It receives the user's request, delegates sub-tasks, collects results, and synthesizes a final answer.
4.  **The HITL Interrupt**: Highlight the pause node. Explain that for high-stakes actions (e.g., "Execute the trade"), the graph **freezes** and serializes its state to the database. A human must explicitly resume it via an API call or UI button.
5.  **MCP**: Briefly mention the **Model Context Protocol** as the direction the industry is moving — a standardized plug-and-play way for agents to discover tools from servers (like Slack, GitHub, etc.) without any custom glue code.

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins

1.  **Open `code/d058-mcp-and-multi-agent-patterns.py`**.
2.  **Worker Agents**:
    - Show the `research_agent` and `finance_agent` with their dedicated tools and specialized system prompts.
3.  **The Supervisor Graph**:
    - Walk through the `StateGraph`. Show how the Router node decides which worker to call.
    - Show `interrupt_before=["verify_action"]` — this is where the HITL pause happens.
4.  **Execution — Session 1 (Paused)**:
    - Run the first invoke. Show how the graph stops at the interrupt node and the state is saved.
5.  **Execution — Session 2 (Resume)**:
    - Pass `None` as input with the same `thread_id`. Show that the graph picks up exactly where it left off.

## Summary Checklist for Trainees
- [ ] Do I understand the difference between the Router and Supervisor patterns?
- [ ] Do I understand how `interrupt_before` freezes the graph and where state is saved?
- [ ] *Extension*: Research how MCP Servers work at `modelcontextprotocol.io`.
