# c266: Advanced Agentic Design Patterns

## Deep Memory Management
As enterprise agents grow more complex, ephemeral buffer memory becomes insufficient. We must rely on structured memory APIs to handle multi-stage conversations and asynchronous interventions.

### The `Store` Interface
LangGraph provides a specialized `Store` interface intended to separate shared application-level memory from individual conversational state. `Store` interactions execute distinct from thread checkpoints, allowing an agent to fetch generalized information—such as company policies or system statuses—independently of what a specific user just typed.

### Cross-Thread Memory
When a single user opens multiple tabs or initiates disparate tasks (threads) across different days, Cross-Thread Memory enables an agent to recognize global continuity. By maintaining identity keys across varying `thread_id` instantiations, the LLM can query the root memory matrix and maintain personalizations globally.

### Namespaces & Scopes (Storing User Preferences)
To properly compartmentalize the `Store` interface across thousands of concurrent users, we utilize logical Namespaces & Scopes:
- **Namespaces**: Partition memory across logical limits (e.g. `['user_preferences', 'user_123']`).
- **Scopes**: Isolate data safely so an LLM processing User A's input cannot accidently inject or retrieve data residing within User B's namespace.

## Advanced Reasoning Patterns
Traditional RAG workflows are inherently reactionary. Advanced systems implement proactive reasoning logic before surfacing results to humans.

### Plan-and-Execute Agents
A Plan-and-Execute architecture splits an agent's brain into two distinct LangGraph nodes:
1. **Planner Node**: Synthesizes the initial goal into a step-by-step array of actionable items.
2. **Executor Node/Agent**: Sequentially iterates through the execution array, triggering specific tools or calculations, before feeding terminal results back to a synthesis component.

### Critique Nodes & Prompts
To counter hallucinations and logic failure, we weave Critique Nodes directly into our graphs. When a sub-system issues a response, it is instantly routed to an isolated LLM evaluator armed with strict grading rubrics (Critique Prompts).

## Asynchronous Architecture
Migrating synchronous graphs up towards performant async configurations is mandatory for heavy production loads.

### Async Operations (`ainvoke`)
Rather than freezing server workers via the standard `.invoke()` endpoint, modern codebases utilize `.ainvoke()`. Coupled with asynchronous checkpointers natively parsing async stream chunks, this allows your REST APIs or WebSocket streams to drastically increase concurrent throughput securely.
