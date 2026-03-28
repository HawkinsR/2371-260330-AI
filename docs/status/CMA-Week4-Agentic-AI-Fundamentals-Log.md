# Weekly Epic: Enter the GenAI era by orchestrating basic reasoning agents and integrating powerful Retrieval Augmented Generation (RAG) capabilities

## 1-Monday

### Written Content

- [x] Create `c256-agentic-design-and-react.md`: Agentic Design Patterns Overview, `init_chat_model` Universal Interface, Chat Models & Temperature, System Prompts vs User Prompts, The `@tool` Decorator & Tool Calling, Structured Output with Pydantic, Simple ReAct Agent Creation (`create_agent`).

### Instructor Demo

- [x] Create `d054-simple-react-agent.py`: Compile a ReAct agent, configure basic `@tool` decorators, and demonstrate deterministic outputs using Pydantic.

### Trainee Exercise

- [x] Create `e040-building-a-basic-agent.md`: Build a simple assistant with an initialized chat model that routes complex calculations to custom `@tool` functions.

## 2-Tuesday

### Written Content

- [x] Create `c257-context-engineering-and-langsmith.md`: Context Engineering Implementation, Dynamic System Prompts (`@dynamic_prompt`), LangChain Standard Application Structure, Tracing Agent Execution, Debugging Traces & Latency, Creating Datasets in LangSmith, Handling Context Window Limits.

### Instructor Demo

- [x] Create `d055-dynamic-prompts-and-tracing.py`: Set up dynamic prompt chaining and trace token outputs natively through LangSmith to monitor latency limits.

### Trainee Exercise

- [x] Create `e041-tracing-agent-execution.md`: Engineer robust dynamic prompts, execute edge-case logic, and evaluate token costs programmatically in LangSmith datasets.

## 3-Wednesday

### Written Content

- [x] Create `c258-vector-databases-and-pinecone.md`: Vector Database Introduction, Embeddings Models & Dimensions, Cosine Similarity vs Euclidean Distance, Pinecone Setup & Configuration, Index Management & Namespaces, CRUD Operations (Upsert/Query), Metadata Filtering Strategies.

### Instructor Demo

- [x] Create `d056-pinecone-upsert-and-query.py`: Configure Pinecone cloud vectors with namespaces, compute embeddings, and execute precise similarity searches with metadata filtering.

### Trainee Exercise

- [x] Create `e042-vector-search-implementation.md`: Embed diverse string datasets and construct logical Pinecone queries implementing dense-vector similarity and namespace isolation.

## 4-Thursday

### Written Content

- [x] Create `c259-document-loaders-and-retrievers.md`: Document Loaders (PDF, Web, Text), Text Splitting Techniques, `create_retriever` Implementation, Vector Search vs Keyword Search, Custom Retriever Logic, Handling Multi-Modal Data, Indexing Optimization.

### Instructor Demo

- [x] Create `d057-custom-retriever-logic.py`: Load generic unformatted text, divide into manageable chunk boundaries, and link directly to a LangChain retriever pipeline.

### Trainee Exercise

- [x] Create `e043-building-a-document-loader.md`: Develop recursive character string splitters on PDF documents, storing chunks securely to be utilized within an indexing retriever stream.

## 5-Friday

### Written Content

- [x] Create `c260-agentic-rag-integration.md`: Retrievers as Tools: Design Pattern, Agentic RAG Workflow, Citations & Provenance, Building a Doc QA Bot (End-to-End).

### Instructor Demo

- [x] Create `d058-doc-qa-bot-end-to-end.py`: Transform the custom retriever into a callable Tool and inject robust citation references alongside the final response payload.

### Trainee Exercise

- [x] Create `e044-agentic-rag-workflow.md`: Construct an end-to-end Doc QA interaction loop answering domain-specific inquiries directly sourced back into their textual origins.
