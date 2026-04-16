# Weekly Epic: Enter the GenAI era by orchestrating basic reasoning agents and integrating powerful Retrieval Augmented Generation (RAG) capabilities

## 1-Monday: LangChain Foundations

### Written Content

- [x] Create `c256-langchain-foundations-and-agents.md`: Build a Basic LangChain Agent with Bedrock Runtime API, `init_chat_model` Universal Interface, System Prompts, Structured Output & Pydantic, Streaming Responses & Token Economics, Context vs. Prompt Engineering.

### Instructor Demo

- [x] Create `d054-langchain-bedrock-foundations.py`: Demonstrate basic agent creation using AWS Bedrock as the runtime, configuring streaming response payloads, and enforcing structured JSON outputs.

### Trainee Exercise

- [x] Create `e040-building-your-first-bedrock-agent.md`: Build a simple assistant using Bedrock that routes complex calculations to custom `@tool` functions and supports real-time streaming.

## 2-Tuesday: Middleware and State Persistence

### Written Content

- [x] Create `c257-middleware-and-memory-persistence.md`: Use Prebuilt and Custom Middleware, Trimming & Sliding Windows, Short-Term vs. Long-Term Memory, Dynamic Few-Shotting Strategies, Injection & Stuffing Guardrails, Multi-Agent Context Handoffs, State Persistence & Step-wise Scratchpads.

### Instructor Demo

- [x] Create `d055-middleware-and-state-management.py`: Implement custom middleware for PII masking and security guardrails while saving agent state using a persistent step-wise scratchpad.

### Trainee Exercise

- [x] Create `e041-securing-and-persisting-agents.md`: Engineer robust memory management by implementing sliding window message history and a persistent state saver with security middleware.

## 3-Wednesday: Vector DBs and Context Optimization

### Written Content

- [x] Create `c258-vector-dbs-and-context-optimization.md`: Vector Database Lifecycle (Pinecone), Serverless vs Pod-based, Index Management, Namespaces, Context Compression & Distillation, Re-ranking & Context Pruning, Observability with LangSmith.

### Instructor Demo

- [x] Create `d056-pinecone-vector-management.py`: Demonstrate live Pinecone CRUD operations (Upsert/Query) and optimize retrieval throughput using Re-ranking and Context Pruning.

### Trainee Exercise

- [x] Create `e042-optimizing-vector-retrieval.md`: Embed datasets into Pinecone while implementing context compression and re-ranking to improve search precision and reduce token noise.

## 4-Thursday: Retrieval Evaluation and RAGAS

### Written Content

- [x] Create `c259-retrieval-evaluation-and-ragas.md`: Word Embeddings & Similarity Search, Distance Metrics, Sentence Transformers & Partitioning, `create_retriever` Implementation, Retrieval Evaluation with RAGAS, Indexing Optimization.

### Instructor Demo

- [x] Create `d057-ragas-retrieval-audit.py`: Build a production LangChain retriever and perform a formal audit of its performance (faithfulness, relevancy) using the RAGAS framework.

### Trainee Exercise

- [x] Create `e043-evaluating-complex-rag.md`: Load and chunk PDF documents, store them in Pinecone, and perform a formal RAGAS evaluation on the results with observability feedback loops.

## 5-Friday: Advanced Agentic Design & Security

### Written Content

- [x] Create `c260-advanced-agents-and-mcp.md`: Prompt Management & Versioning, Online Evaluation & HITL, Multi-Agent Orchestration, Model Context Protocol (MCP), Injection & Stuffing Guardrails, Context Engineering.

### Instructor Demo

- [x] Create `d058-mcp-and-multi-agent-patterns.py`: Demonstrate a production multi-agent system using MCP for data interaction and implementing Human-in-the-Loop (HITL) approval workflows.

### Trainee Exercise

- [x] Create `e044-production-grade-agent-security.md`: Construct a production-grade secured bot that implements multi-agent handoffs, MCP-based tool discovery, and advanced context engineering.
