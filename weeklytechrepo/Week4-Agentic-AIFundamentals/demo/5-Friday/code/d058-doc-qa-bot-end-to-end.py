"""
Demo: End-to-End Agentic RAG (Retrievers as Tools)
This script demonstrates how to integrate a static Vector Database completely inside
a dynamic ReAct agent loop. It gives the LLM the autonomy to decide WHEN to search 
the database, and forces it to cite its sources when it does.
"""

from typing import TypedDict, List
import json

# =====================================================================
# 1. Simulating the Retreiver
# =====================================================================
# We simulate calling a Pinecone/FAISS index that returns documents.
def mock_retriever_search(query: str) -> List[dict]:
    """Simulates vectorstore.as_retriever().invoke()"""
    print(f"\n   [Vector Database] Similarity search executing for: '{query}'")
    
    # A hardcoded simulation of Cosine Similarity returning the top chunks
    if "pto" in query.lower() or "vacation" in query.lower():
        return [
            {"content": "Full-time employees receive 15 days of PTO per calendar year.", "meta": {"source": "handbook_page_4.pdf"}},
            {"content": "Up to 5 days of PTO can roll over to the next year.", "meta": {"source": "handbook_page_5.pdf"}}
        ]
    elif "reimburse" in query.lower() or "expense" in query.lower():
        return [
            {"content": "All travel expenses must be submitted within 30 days via Workday.", "meta": {"source": "finance_policy.pdf"}}
        ]
    return []

# =====================================================================
# 2. Wrapping the Retriever as an Agentic Tool
# =====================================================================
# In LangChain, this would be `create_retriever_tool(retriever, name, desc)`
def tool_search_corporate_knowledge(query: str) -> str:
    """
    Search the internal corporate knowledge base for HR, Finance, and IT policies.
    """
    print(f"\n   [Tool Executing] 'search_corporate_knowledge' triggered by Agent")
    # The tool calls the database
    docs = mock_retriever_search(query)
    
    if not docs:
        return "Search returned no results."
        
    # Format the documents so the LLM can read both the text AND the origin metadata
    formatted_docs = ""
    for idx, doc in enumerate(docs):
        # We append the exact string and its source file so the LLM can see where it came from
        formatted_docs += f"--- Document {idx+1} ---\nContent: {doc['content']}\nSource: {doc['meta']['source']}\n\n"
        
    # We return a single massive string back to the LLM's short-term memory
    return formatted_docs

# =====================================================================
# 3. The Agentic RAG Loop
# =====================================================================
def simulate_agentic_rag(user_prompt: str) -> dict:
    """
    Simulates the LangGraph wrapper (create_react_agent) integrating the 
    Retriever Tool alongside rigid citation rules.
    """
    print(f"\n[USER PROMPT]: '{user_prompt}'")
    print("-" * 50)
    
    # --- The ReAct Decision Tree ---
    
    # 1. Does the prompt require corporate tools?
    # This simulates the LLM reasoning about whether to use the retriever or not
    requires_internal_knowledge = any(keyword in user_prompt.lower() for keyword in ["pto", "vacation", "reimburse", "expense", "policy"])
    
    if not requires_internal_knowledge:
        # Standard ChatGPT path (No RAG needed)
        print("AGENT THOUGHT: This is a general knowledge question. I do not need to query the corporate database.")
        print("AGENT ACTION: Generating standard response natively.")
        final_answer = "I am happy to help with general questions! However, let me know if you need to know about internal corporate policies."
        citations = []
        
    else:
        # Agentic RAG path
        print("AGENT THOUGHT: The user is asking about internal corporate policies. I need to search the knowledge base.")
        print("AGENT ACTION: Calling 'search_corporate_knowledge' tool...")
        
        # Agent calls the tool passing over its search query
        tool_output = tool_search_corporate_knowledge(user_prompt)
        print("AGENT OBSERVATION: Received documents from database.")
        
        print("AGENT THOUGHT: I will extract the exact answer from these documents and explicitly log the 'Source' metadata to prevent hallucination claims.")
        
        # Simulating the LLM synthesizing the tool output into a final cohesive response
        if "pto" in user_prompt.lower():
            final_answer = "You receive 15 days of PTO per calendar year, and you are permitted to roll over up to 5 days into the following year."
            # The LLM explicitly references the metadata it saw in the tool output
            citations = ["handbook_page_4.pdf", "handbook_page_5.pdf"]
        elif "expense" in user_prompt.lower():
            final_answer = "You must submit all travel expenses within 30 days of the trip using the Workday system."
            citations = ["finance_policy.pdf"]
        else:
            final_answer = "I could not find the answer to that in the policy documents."
            citations = []

    # Final Payload Assembly returning a structured object with guaranteed citations
    return {
        "answer": final_answer,
        "citations": citations
    }

# =====================================================================
# 4. End-to-End Execution
# =====================================================================
def run_agentic_rag_demo():
    print("=== Agentic AI Fundamentals: End-to-End Doc QA Bot ===")
    
    # Scenario A: RAG explicitly required for internal knowledge
    print("\n\n>>> SCENARIO A: Corporate Query <<<")
    result_a = simulate_agentic_rag("How many PTO days can I roll over to next year?")
    print(f"\n[FINAL RESPONSE]: {result_a['answer']}")
    print(f"[PROVENANCE CITATIONS]: {result_a['citations']}")
    
    # Scenario B: Basic query (Standard Chat doesn't waste database compute)
    print("\n\n>>> SCENARIO B: General Query <<<")
    result_b = simulate_agentic_rag("Why is the sky blue?")
    print(f"\n[FINAL RESPONSE]: {result_b['answer']}")
    print(f"[PROVENANCE CITATIONS]: {result_b['citations']}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    run_agentic_rag_demo()
