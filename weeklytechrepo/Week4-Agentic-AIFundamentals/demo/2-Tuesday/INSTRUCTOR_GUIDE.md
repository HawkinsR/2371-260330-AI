# Instructor Guide: Middleware and State Persistence

## Overview
This demo demonstrates how to add **Reliability** and **Security** to an agentic workflow using **LangGraph Persistence** and custom message interception.

## Phase 1: The Concept (Whiteboard)
**Time:** 10 mins

1.  **Open `diagrams/middleware_flow.mermaid`**.
2.  **State Persistence**: Explain that LLMs are stateless by default. To have a real conversation, we must save the "thread" to a database.
3.  **Middleware**: Show how we can "intercept" the message *before* it gets to the LLM (for security) and *after* the LLM responds (for logging).
4.  **Discussion**: Ask: "What happens if our server crashes mid-thought?" (Answer: Without persistence, the thread is lost. With checkpointers, the agent can resume from the exact node it was on).

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins

1.  **Open `code/d055-middleware-and-state-management.py`**.
2.  **The Checkpointer**:
    - Highlight the `SqliteSaver`. Explain that this is a production-style pattern (though we'd use Postgres in a massive enterprise).
    - Point out the `thread_id` and how it acts as the primary key for the session.
3.  **The Middleware Node**:
    - Show how we create a simple function that scans the state's `messages` and redacts PII.
    - Reinforce that this happens *outside* the LLM's reasoning to save tokens and improve security.
4.  **Resumption**:
    - Call the agent once, then call it again with the *same* `thread_id`.
    - Show that the agent "remembers" context without us manually passing it in the second call.

## Summary Checklist for Trainees
- [ ] Does my `thread_id` stay consistent across the session?
- [ ] Is the `agent_memory.db` file being created in the local directory?
- [ ] Does the PII masking work before the prompt is sent to Bedrock?
