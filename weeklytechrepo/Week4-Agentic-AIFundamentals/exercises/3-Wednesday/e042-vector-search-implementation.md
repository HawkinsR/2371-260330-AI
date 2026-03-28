# Lab: Vector Search Implementation

## The Scenario
Your company is building an internal Knowledge Base chatbot. However, a traditional SQL keyword search keeps failing because employees use different words than the official documentation (e.g., asking for "days off" instead of "PTO"). You have been tasked with taking raw string documents, converting them into mathematical vectors (embeddings), and storing them securely within an isolated cluster (Namespace) inside a Pinecone vector database. Finally, you must prove the system works by executing a semantic query on the database and retrieving the correct document.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e042-vector_search_lab.py`.
3. Complete the `ingest_knowledge_base` function:
   - Convert the string `doc_text` into a vector array by passing it to the provided `get_mock_embedding()` function.
   - Construct the list of records to upsert. You must include the `id`, `values` (the vector you just created), and a `metadata` dictionary containing at minimum `{"text": doc_text}`.
   - Call `.upsert()` on the `db` object. Pass your records and explicitly set the `namespace` to `"hr-policies"`.
4. Complete the `retrieve_answer` function:
   - Convert the `user_question` into a query vector using `get_mock_embedding()`.
   - Call `.query()` on the `db` object. 
   - Pass your query `vector`, request the top `1` result (`top_k=1`), constrain the search to the `"hr-policies"` `namespace`, and optionally filter by metadata if necessary.
   - Return the highest scoring match text.

## Definition of Done
- The script executes successfully and prints the UPSERT log for the `hr-policies` namespace.
- The user query `"I am feeling sick today, who do I call?"` correctly retrieves the HR medical leave policy document, even though the query and the document share very few keywords.
