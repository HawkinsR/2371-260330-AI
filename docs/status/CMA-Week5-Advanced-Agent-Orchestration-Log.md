# Weekly Epic: Upgrade AI reasoning logic by integrating LangGraph workflows, routing conditional logic, and employing multi-agent architectures

## 1-Monday

### Written Content

- [x] Create `c261-advanced-retrieval-and-evaluation.md`: Advanced Retrieval Patterns, Query Decomposition & Multi-Query Retriever Pattern, Self-Querying & Metadata Construction, NER for Metadata Extraction, Hybrid Search (Sparse + Dense), Re-ranking Retrieved Results, Retrieval Evaluation, RAG Evaluation with RAGAS, Context Precision & Recall.

### Instructor Demo

- [x] Create `d059-multi-query-and-hybrid-search.py`: Configure sparse/dense hybrid searches coupled with a multi-query retriever and apply basic RAGAS evaluation over precision.

### Trainee Exercise

- [x] Create `e045-evaluating-rag-with-ragas.md`: Break down queries via decomposition logic, evaluate context recall using RAGAS mechanisms, and apply simple reranking.

## 2-Tuesday

### Written Content

- [x] Create `c262-langgraph-state-and-routing.md`: `StateGraph` vs `create_agent`, Defining Graph State (`TypedDict`), Nodes & Edges Basics, Conditional Edges & Routing, Compiling the Graph, LangSmith Studio Prototyping, Handling Graph Errors.

### Instructor Demo

- [x] Create `d060-stategraph-compilation.py`: Differentiate LangChain and LangGraph by creating an explicit `TypedDict` state, connecting distinct nodes via static and conditional routing edges.

### Trainee Exercise

- [x] Create `e046-conditional-routing-graph.md`: Map an acyclic condition tree to direct inputs depending on graph-level heuristics using a freshly compiled `StateGraph`.

## 3-Wednesday

### Written Content

- [x] Create `c263-runtime-configuration-and-middleware.md`: Runtime Configuration (`configurable`), Passing State to Tools (`ToolRuntime`), Middleware Patterns (`@wrap_model_call`), Managing Conversation History, Trimming Messages for Context, Shared State across Nodes.

### Instructor Demo

- [x] Create `d061-toolruntime-and-state-management.py`: Handle internal tool runtime contexts and trim older conversational messages before routing them dynamically across parallel nodes.

### Trainee Exercise

- [x] Create `e047-managing-conversation-history.md`: Design configurable global state wrappers adjusting token context window footprints using recursive message trimming heuristics.

## 4-Thursday

### Written Content

- [x] Create `c264-orchestrator-workers-architecture.md`: Orchestrator-Workers Architecture, Defining Sub-Agent Interfaces, Routing Tasks to Sub-Agents, Aggregating Sub-Agent Outputs, Handoffs between Agents, Managing Sub-Agent Context, Supervisor Implementation.

### Instructor Demo

- [x] Create `d062-supervisor-agent-implementation.py`: Formulate an orchestrator layer that accepts generalized instructions, routes them intelligently to specialized worker sub-agents, and aggregates final states.

### Trainee Exercise

- [x] Create `e048-routing-tasks-to-sub-agents.md`: Develop separate agent logic blocks and unify them under an overarching Supervisor script performing multi-agent sub-routing computations.

## 5-Friday

### Written Content

- [x] Create `c265-breakpoints-and-time-travel.md`: Breakpoints & `interrupt`, `Command` for State Updates, Time Travel (Rewinding/Forking), Approval Workflows, Editing State on the Fly, Streaming Events during Interruption.

### Instructor Demo

- [x] Create `d063-graph-interrupts-and-state-editing.py`: Trigger graph breakpoints mid-execution to intercept tool payloads, edit dynamic variables live, and implement manual approval workflows.

### Trainee Exercise

- [x] Create `e049-approval-workflow-implementation.md`: Fork historical run states asynchronously to execute "time-travel" functionality and test manual Human-in-the-Loop interruptions.
