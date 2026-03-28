# Lab: Building a Document Loader and Splitter

## The Scenario
Your company has a massive Employee Handbook. Before you can upload it to the Vector Database, you must build the data ingestion pipeline. A naive approach of just splitting the document by every 10 words will cut sentences in half, destroying the semantic meaning. You need to simulate a `RecursiveCharacterTextSplitter` to break the document into overlapping chunks, maintaining the context at the boundaries so the Retriever can accurately find answers.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e043-document_loader_lab.py`.
3. Complete the `chunk_document` function:
   - This function receives the `master_text` string, a `chunk_size` (number of words), and a `chunk_overlap` (number of words).
   - Split the `master_text` into a list of words.
   - Iterate over the words using a `while` loop. 
   - Extract a list of words of length `chunk_size`.
   - Join them into a string and create a `Document` object with `page_content` and include `{"chunk_id": len(chunks)}` in the metadata.
   - Importantly, advance your loop counter by `(chunk_size - chunk_overlap)` so the chunks overlap.
   - Return the list of structured `Document` chunks.

## Definition of Done
- The script executes successfully.
- It prints the generated chunks showing the text overlap. 
- The simulated retriever successfully locates Chunk 2 when queried about "hardware delays".
