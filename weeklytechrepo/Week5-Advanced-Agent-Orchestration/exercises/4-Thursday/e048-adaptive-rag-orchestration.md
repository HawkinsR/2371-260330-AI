# E048: Adaptive RAG Orchestration & HITL

## Objective
Build a dynamic router node that classifies queries, combined with a Human-in-the-Loop breakpoint.

## Instructions
1. Open `e048-adaptive-rag.py` inside the `starter_code/` directory.
2. Initialize `ChatBedrock`.
3. Build a router function using Bedrock to evaluate if a user's prompt is a "vector_search" or a "web_search". Do not use static IF string matching like `if 'policy' in query`.
4. Create a graph node that processes the vector search path. Add a LangGraph `interrupt()` call right before the search executes, prompting an instructor for approval.
5. Compile the `StateGraph` using a `MemorySaver` checkpointer (required for interrupts).
6. Stream a prompt and ensure the execution pauses securely.
