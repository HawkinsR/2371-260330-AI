# E045: Querying the Index and Namespace

## Objective
Configure a `FAISS` vector store utilizing AWS `BedrockEmbeddings`. Set up LangSmith environment variables locally to trace the semantic retrieval execution. 

## Instructions
1. Navigate to the `starter_code/` directory and open `e045-querying-index.py`.
2. Ensure you have your `LANGSMITH_API_KEY` exported in your environment.
3. Replace the placeholder embedding model with `langchain_aws.BedrockEmbeddings(region_name="us-east-1")`.
4. Initialize the `FAISS` vector store from the provided dummy documents.
5. Execute a `similarity_search` and verify the traced output appears in your LangSmith project console.
