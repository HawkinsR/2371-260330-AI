# Demo: Document Loaders and Retrievers

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
| --- | --- |
| **Document Loader** | *"What's the difference between a raw `.pdf` file bytes and cleaned text paragraphs ready for an LLM? What does a document loader do in that gap?"* |
| **Chunking** | *"Why can't you embed an entire 500-page book as one single vector and call it done? What information do you lose with a vector that represents 500 pages vs. 500 separate paragraph-sized chunks?"* |
| **Chunk Overlap** | *"A sentence reads: 'The policy applies to full-time employees. They are entitled to 15 days.' If the chunk splits after the first sentence, who does 'They' refer to? How does overlap fix this?"* |
| **Retriever as Abstraction** | *"If today you use FAISS as your vector store but next month you migrate to Pinecone, what part of your code breaks? How does the `as_retriever()` abstraction protect the rest of your pipeline from this change?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/document-loaders-retrievers.mermaid`.
2. Trace the Data Ingestion pipeline. Start at Data Loading. Emphasize that PDF, TXT, and HTML must all be converted into a universal `Document` class containing `page_content` and `metadata`.
3. Explain the **Chunking Phase**. Ask: "If my context window is 8k tokens, why not just make my chunk size 8k?" (Answer: If you feed 8k tokens of irrelevant fluff alongside the 1 sentence of actual data, the LLM will become distracted and hallucinate. High precision requires smaller chunks!).
4. Explain **Chunk Overlap**. "If the text is 'Apple released the iPhone 15. It costs $999.', what happens if the chunk splits exactly between the two sentences?" The pronoun 'It' loses its antecedent. An overlap of 1-2 sentences prevents this data loss at the seams.
5. Emphasize the `vectorstore.as_retriever()` wrapper. It acts as an abstraction layer so the rest of the application never has to know whether it's querying FAISS, Pinecone, or ChromaDB.

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d057-custom-retriever-logic.py`.
2. Review Phase 1 and Phase 2. The `simulate_recursive_text_splitter` breaks the giant string into small chunks by word count (or characters).
3. Execute the script via `run_retriever_pipeline()`. 
4. Stop at the `[Debug] Inspecting Chunk Overlap:` section. Visually show the students how Chunk 0 ends with "Costs were significantly" and Chunk 1 *starts* with "Costs were significantly". Trace how the data overlaps to ensure no semantic bridges are destroyed.
5. Review Phase 3 and 4. Show the `[Retriever] Executing search`. A query about saving money bypassed Chunks 0 and 2, and accurately narrowed in explicitly on Chunk 1, returning its exact `page_content` to serve as pure, dense context to give to the LLM.

## Summary
Summarize that an LLM can only reason effectively if we feed it pristine, dense data. Messy text extraction and poor chunking parameters will destroy RAG performance regardless of how powerful the Vector Database or LLM is. Clean Data = Clean RAG.
