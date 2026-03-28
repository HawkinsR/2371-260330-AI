# Lab: End-to-End Agentic RAG Workflow

## The Scenario
Your standard RAG pipeline is wasting expensive API calls querying the database every time a user simply says "Hello." You have been tasked with upgrading the system to an Agentic RAG architecture. You must wrap the Retriever inside a callable Tool (`tool_search_corporate_knowledge`) and place it inside an agent reasoning loop. The agent must autonomously decide *when* to search the knowledge base. Furthermore, to prevent hallucination claims, you must enforce a strict citation policy extracting the `source` metadata from any retrieved documents and returning it in the final payload.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e044-agentic_rag_lab.py`.
3. Complete the `tool_search_corporate_knowledge` function:
   - Call the `mock_retriever_search` function passing the `query`.
   - Iterate through the returned `docs`.
   - Build and return a formatted string that includes both the `content` of the document and the `source` metadata so the LLM can read both. If no docs, return `"Search returned no results."`
4. Complete the `simulate_agentic_rag` function:
   - Implement the decision logic. Check if the `user_prompt` requires internal knowledge (e.g., contains keywords like `pto`, `vacation`, `reimburse`, `expense`, or `policy`).
   - If internal knowledge is NOT required, return a standard greeting and an empty `citations` list.
   - If internal knowledge IS required, call your `tool_search_corporate_knowledge` to retrieve the `tool_output`. Extract the specific answer based on the `user_prompt` and populate the `final_answer` and the `citations` list with the appropriate `source` metadata.

## Definition of Done
- The script executes successfully.
- Scenario A outputs a specific answer accurately citing the `handbook_page_4.pdf` and `handbook_page_5.pdf` documents.
- Scenario B outputs a standard conversational response gracefully bypassing the Vector Database entirely.
