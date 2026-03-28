# Weekly Epic: Ready pipelines for enterprise deployment natively adopting iterative memory stores, dynamic safety guards, and async streaming protocols

## 1-Monday

### Written Content

- [x] Create `c266-persistent-memory-and-checkpointers.md`: Short-term vs Long-term Memory, Persistent Checkpointers for Sessions, The `Store` Interface, Namespaces & Scopes, Storing User Preferences, Cross-Thread Memory, RAG vs Parametric vs Ephemeral Memory.

### Instructor Demo

- [x] Create `d064-cross-thread-memory-store.py`: Initialize thread-persistent checkpointers using localized SQLite engines, committing distinct scopes dynamically back to the `Store`.

### Trainee Exercise

- [x] Create `e050-implementing-long-term-memory.md`: Craft functional interfaces isolating unique memory footprints across disparate users simulating a live multi-tenant AI.

## 2-Tuesday

### Written Content

- [x] Create `c267-self-correction-and-plan-and-execute.md`: Self-Correction Loops, Plan-and-Execute Agents, Critique Nodes & Prompts, Iterative Refinement Patterns, Output/Input Validation Guards, Handling Hallucinations with Reflection.

### Instructor Demo

- [x] Create `d065-iterative-refinement-agent.py`: Construct self-corrective loops where output verification nodes critique a primary output layer pushing iterations till standards format alignment occurs.

### Trainee Exercise

- [x] Create `e051-plan-and-execute-validation.md`: Engineer a planning-executor graph enforcing strict Pydantic validity, handling fallback hallucination responses with automated retry hooks.

## 3-Wednesday

### Written Content

- [x] Create `c268-evaluation-driven-development.md`: Evaluation Driven Development (EDD), Creating Golden Datasets, Unit Testing Agents, Mocking Tool Calls, Feedback Loops & Annotations, Custom Evaluators in LangSmith.

### Instructor Demo

- [x] Create `d066-mocking-tools-and-unit-testing.py`: Unit test a LangGraph function abstracting tool calls with patched constants and establishing comprehensive standard testing pipelines visually against LangSmith Golden Datasets.

### Trainee Exercise

- [x] Create `e052-creating-golden-datasets.md`: Curate offline evaluation heuristics matching agent outputs statically against preset annotations establishing functional unit coverage benchmarks.

## 4-Thursday

### Written Content

- [x] Create `c269-langgraph-cloud-deployment.md`: Deploying to LangGraph Cloud, Async Operations (`ainvoke`), Streaming Events (v2).

### Instructor Demo

- [x] Create `d067-async-operations-and-streaming.py`: Update legacy invoke mechanisms securely migrating synchronous blocking graphs up towards performant async `ainvoke` configurations parsing stream chunks sequentially.

### Trainee Exercise

- [x] Create `e053-deploying-agent-to-cloud.md`: Package dependencies, structure deployment manifests dynamically, and upload execution graphs seamlessly scaling onto the localized cloud layer.

## 5-Friday

### Written Content

- [x] Create `c270-security-and-capstone-project.md`: Security Guardrails, Prompt Injection Defense, Final Capstone Project Presentation.

### Instructor Demo

- [x] Create `d068-prompt-injection-defense.py`: Introduce malicious input sequences simulating external prompt injections and implement defensive filters sanitizing prompt leakage.

### Trainee Exercise

- [x] Create `e054-final-capstone-presentation.md`: Review overarching progress, synthesize learning into a holistic pipeline format showcasing integrated LangGraph features with complete robust RAG capabilities alongside safety standards.
