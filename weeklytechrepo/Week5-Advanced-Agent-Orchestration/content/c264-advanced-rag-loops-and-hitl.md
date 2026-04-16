# c264: Advanced RAG Loops and Human in the Loop

## Advanced RAG Agents
Traditional RAG builds a linear context window. Advanced Agentic RAG transforms this into a dynamic, cyclical workflow:
- **Agentic RAG**: The agent decides if it needs to search, execute multiple discrete searches, or refine its queries.
- **Adaptive RAG**: Uses a router node to categorize a query and decide the strategy (e.g., direct answer, vector search, or web search).
- **Corrective RAG (CRAG)**: Evaluates retrieved documents. Irrelevant context is discarded, and the agent falls back to alternative data sources.
- **Self-RAG**: The model generates an answer and then critiques its own generation against the retrieved context to ensure zero hallucination.

## Human in the Loop (HITL)
Breakpoints pause the execution of an agentic workflow to allow for human interaction or manual review, effectively creating Human-in-the-Loop workflows.

### Approval Workflows
In sensitive scenarios (like executing financial transactions or sending external emails), the graph pauses on a tool call, waiting for an explicit "Approve" or "Reject" signal from a human user.

## Editing State and Time Travel
- **Editing State**: During a breakpoint, a human operator can modify the graph's internal state on the fly before resuming the execution.
- **Time Travel**: Because LangGraph uses persistent checkpoints for every node transition, you can rewind the thread history to a previous step, edit the state, and fork the execution into a new path.
