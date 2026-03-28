# Demo: Agentic RAG Integration

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
| --- | --- |
| **Standard RAG vs. Agentic RAG** | *"In Standard RAG, the retrieval happens every single time without exception. What's the wasteful scenario that creates? Give me an example of a user query where retrieval adds zero value."* |
| **Provenance / Citation** | *"A chatbot tells a nurse: 'Patients can be safely given 500mg of drug X.' The claim is wrong. The model hallucinated it. Why is knowing exactly which document the model retrieved crucial for liability? What does provenance provide?"* |
| **Tool as Autonomous Decision** | *"If the retriever is a tool, the LLM decides whether to call it. What criteria do you think the agent uses to decide 'I need to search the database for this' vs. 'I can answer this from general knowledge'?"* |
| **End-to-End AI Architecture** | *"Trace the entire journey: a user asks a question, and 5 seconds later they get a cited answer. Name every major component that ran in those 5 seconds, in order."* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/agentic-rag-workflow.mermaid`.
2. Trace the loop. Ask the class: "What is the primary difference between Standard RAG and Agentic RAG?" 
3. *Answer:* Standard RAG creates a hardcoded pipeline. EVERY user prompt triggers a database search blindly. In **Agentic RAG**, the database is just another Tool on the LLM's toolbelt (`create_retriever_tool`). The LLM acts as the routing brain. It can decide to perform Math, or Web Search, or Database Search, or just answer the question natively. 
4. Explain **Citations**. If a chatbot tells an employee they get 50 vacation days, and the employee sues the company, who is at fault? By forcing the LLM to output the `Document.metadata['source']` alongside its answer, we establish absolute provenance for auditing LLM accuracy.

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d058-doc-qa-bot-end-to-end.py`.
2. Review `tool_search_corporate_knowledge()`. Show how we take the raw Vector Search results and package them into a simple, formatted string containing both the `Content` and the `Source` metadata. This string becomes the "Observation" the agent reads.
3. Review `simulate_agentic_rag()`. Explain that we dictate the agent's behavior via its System Prompt (represented here by the rigid decision tree in code). 
4. Execute the script via `run_agentic_rag_demo()`. 
5. Trace **Scenario A**. The agent recognizes it needs policy data. It executes the tool. It synthesizes the returned chunks, and crucially, it outputs a strict JSON payload containing the answer AND the exact PDFs the data came from.
6. Trace **Scenario B**. The user asks about the sky. The agent recognizes this is general knowledge. *It bypassed the database entirely*. It did not waste vectors, latency, or compute tokens performing a semantic math search for the color of the sky in the HR handbook.

## Summary
Reiterate that compiling retrievers into Tools completes the evolution from prompt-engineers to AI architects. We are no longer building chatbots; we are building autonomous software operators that can securely query multi-dimensional databases, audit their own references, and synthesize enterprise intelligence on demand.
