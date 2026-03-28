from typing import List

# =====================================================================
# MOCK LLM & DATABASE (Do not edit this section)
# =====================================================================
def mock_retriever_search(query: str) -> List[dict]:
    print(f"\n   [Vector DB] Search executing for: '{query}'")
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
# YOUR TASKS
# =====================================================================
def tool_search_corporate_knowledge(query: str) -> str:
    print(f"\n   [Tool] 'search_corporate_knowledge' triggered")
    docs = mock_retriever_search(query)
    
    if not docs:
        return "Search returned no results."
        
    formatted_docs = ""
    
    # 1. TODO: Iterate over the docs. 
    # Create a formatted string displaying the document index, content, and the source metadata.
    
    
    return formatted_docs

def simulate_agentic_rag(user_prompt: str) -> dict:
    print(f"\n[USER PROMPT]: '{user_prompt}'")
    
    # --- The ReAct Decision Tree ---
    # 2. TODO: Check if the user prompt requires internal knowledge.
    # Look for keywords: pto, vacation, reimburse, expense, policy
    requires_internal_knowledge = False 
    
    if not requires_internal_knowledge:
        print("AGENT: This is a general question. Bypassing Vector DB.")
        
        # 3. TODO: Set basic answer and empty citations
        final_answer = ""
        citations = []
        
    else:
        print("AGENT: Internal policies required. Calling database tool...")
        
        # 4. TODO: Call the tool_search_corporate_knowledge tool
        tool_output = ""
        
        print("AGENT: Synthesizing tool output into final answer with citations...")
        
        # 5. TODO: Formulate the final answer and populate citations based on the prompt
        if "pto" in user_prompt.lower():
            final_answer = ""
            citations = []
        elif "expense" in user_prompt.lower():
            final_answer = ""
            citations = []
        else:
            final_answer = "Policy not found."
            citations = []

    return {
        "answer": final_answer,
        "citations": citations
    }

# =====================================================================
# 4. End-to-End Execution
# =====================================================================
if __name__ == "__main__":
    print("=== Agentic AI: Chat QA Bot ===")
    
    print("\n>>> SCENARIO A: Corporate Query <<<")
    result_a = simulate_agentic_rag("How many PTO days can I roll over to next year?")
    print(f"\n[FINAL RESPONSE]: {result_a['answer']}")
    print(f"[CITATIONS]: {result_a['citations']}")
    
    print("\n>>> SCENARIO B: General Query <<<")
    result_b = simulate_agentic_rag("Why is the sky blue?")
    print(f"\n[FINAL RESPONSE]: {result_b['answer']}")
    print(f"[CITATIONS]: {result_b['citations']}")
