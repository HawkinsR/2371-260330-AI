# Weekly Epic: Upgrade AI reasoning logic by integrating LangGraph workflows, routing conditional logic, and employing multi-agent architectures

## 1-Monday
### Written Content
- [ ] Create `c261-semantic-search-and-metadata.md`: LangSmith Documentation, Vector Query with Semantic Search, Named Entity Recognition (NER), Vector Embedding in Batches, Querying the Index and Namespace, Performance Optimization, Metadata Filtering, Backup and Collections Overview.

### Instructor Demo
- [ ] Create `d059-vector-search-and-metadata.py`: Demonstrate Vector Embedding in Batches, semantic search, and metadata filtering.

### Trainee Exercise
- [ ] Create `e045-querying-index-and-namespace.md`: Query the Index and Namespace context using semantic search and refer to LangSmith Documentation for observability.

## 2-Tuesday
### Written Content
- [ ] Create `c262-langgraph-fundamentals.md`: Introduction to LangGraph, Agents as Graphs Overview, LangChain vs LangGraph, Nodes, Edges, Graphs, Graph Errors, `StateGraph` vs `create_agent`, `TypedDict` State, Memory, Persistence, Conditional Edges, Binding the tools, Streaming, Command for State Updates, LangGraph Introduction Overview.

### Instructor Demo
- [ ] Create `d060-stategraph-foundations.py`: Build a Basic LangGraph Agent, configure nodes and edges, bind tools, and demonstrate memory and persistence.

### Trainee Exercise
- [ ] Create `e046-langgraph-agents-task.md`: Build a functional `StateGraph` using conditional edges that saves state across sessions and uses `Command` for mid-run state updates.

## 3-Wednesday
### Written Content
- [ ] Create `c263-orchestration-and-hand-off-patterns.md`: Routing/Aggregating/Handoff Patterns, Supervisor, Sub-Agent Interfaces, Orchestrator-Workers Architecture.

### Instructor Demo
- [ ] Create `d061-supervisor-pattern-implementation.py`: Build a Supervisor graph that routes tasks to specialized sub-agents and aggregates their terminal state.

### Trainee Exercise
- [ ] Create `e047-multi-agent-system-handoffs.md`: Build a Multi-Agent RAG System where a "Search" agent hands off findings to an "Analyst" agent for synthesis.

## 4-Thursday
### Written Content
- [ ] Create `c264-advanced-rag-loops-and-hitl.md`: Advanced RAG Agents (Agentic, Adaptive, Corrective and Self-RAG), Human in the loop, Editing State, Time Travel, Approval Workflows.

### Instructor Demo
- [ ] Create `d062-self-rag-and-hitl.py`: Demonstrate a Self-RAG loop and triggering an interrupt for manual human approval (HITL) and Time Travel.

### Trainee Exercise
- [ ] Create `e048-adaptive-rag-orchestration.md`: Implement an Adaptive RAG workflow that incorporates Approval Workflows and Editing State on the fly.

## 5-Friday
### Written Content
- [ ] Create `c265-production-deployment.md`: Deployment Options, LangGraph Platform Overview.

### Instructor Demo
- [ ] Create `d063-lambda-api-gateway/` folder containing `app.py` (Lambda handler), `template.yaml` (SAM IaC), and `deploy.sh` (deployment script): Demonstrate deploying a LangGraph agent workflow to AWS Lambda via API Gateway using AWS SAM.

### Trainee Exercise
- [ ] Create `e049-langgraph-deployment-task.md`: LangGraph Deployment Task.
