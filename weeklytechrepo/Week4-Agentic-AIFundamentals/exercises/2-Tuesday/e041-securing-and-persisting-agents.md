# Exercise e041: Securing and Persisting Agents

## Overview
In this exercise, you will add **production-grade reliability and security** to a LangGraph agent. You'll implement a SQLite-backed checkpointer for state persistence and a PII-masking middleware node that intercepts user messages before they reach the LLM.

## Learning Outcomes
- Use `SqliteSaver` to persist agent state across multiple sessions.
- Implement a custom **middleware graph node** that scans for and redacts PII.
- Understand the `thread_id` pattern for identifying persistent sessions.
- Demonstrate agent "memory" by resuming a conversation without re-passing history.

## Prerequisites
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION` set in your environment.
- `pip install langchain-aws langgraph`

## Instructions

Open `starter_code/e041-persistence-lab.py` and complete the TODOs:

1. **Create the PII Middleware Node** — Write a node function that checks the last `HumanMessage` in state for any 16-digit number patterns (credit cards) and redacts them with `[REDACTED]`.
2. **Wire Up SqliteSaver** — Create a `sqlite3` connection and pass it to `SqliteSaver`. Compile your graph with `checkpointer=memory`.
3. **Add the Interrupt (Bonus)** — Add `interrupt_before=["model"]` to your compile step and observe how the graph pauses for human review.
4. **Demonstrate Resumption** — Run Session 1 to establish a fact, then run Session 2 with the *same* `thread_id` and ask the agent a follow-up that requires the earlier fact.

## Deliverable
Show that the agent answers a follow-up question in Session 2 correctly, without you passing the original context again.
