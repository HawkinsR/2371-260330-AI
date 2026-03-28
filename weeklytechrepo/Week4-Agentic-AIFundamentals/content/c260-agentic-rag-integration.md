# Agentic RAG Integration

## Learning Objectives

- Convert standard search indices into tool-driven interfaces using the Retrievers as Tools Design Pattern.
- Orchestrate the complete Agentic RAG Workflow seamlessly merging reasoning nodes with static retrieval.
- Safeguard LLM hallucination enforcing rigorous Citations & Provenance mapping.
- Compile and execute a comprehensive Doc QA Bot (End-to-End).

## Why This Matters

This is the culmination of Week 4. Standard RAG forces a rigid pipeline: User asks a question -> Search Database -> Pass text to LLM -> LLM answers. But what if the user just says "Hi"? Standard RAG would needlessly query the database for "Hi." By integrating retrieving actions as explicit Tools within an Agentic ReAct loop, the LLM makes autonomous, intelligent decisions. It only searches the database when it actually identifies a gap in its knowledge.

> **Key Term - Agentic RAG:** An evolved form of Retrieval-Augmented Generation where the retrieval step is not hard-coded into the pipeline but instead offered to an AI agent as an optional tool. The agent decides autonomously *whether* and *when* to retrieve documents based on the complexity of the query, rather than always triggering a database search regardless of context.

## The Concept

### Retrievers as Tools

Instead of hooking the retriever directly to the LLM's input stream, we wrap the `retriever.invoke()` function inside an `@tool`. We define a clear description (e.g., "Use this tool to search the employee handbook for HR questions"). The Agent now has a choice: answer standard questions natively, or use the tool to fetch proprietary context.

### The Agentic RAG Workflow

1. **User Input:** "How many PTO days do I get, and what is the weather today?"
2. **Reasoning Loop 1:** The agent identifies it needs HR data. It calls the `HandbookRetriever` tool.
3. **Observation 1:** The tool returns chunked documents explaining that employees get 15 PTO days.
4. **Reasoning Loop 2:** The agent identifies it needs weather data. It calls the `WeatherAPI` tool.
5. **Observation 2:** The tool returns "Sunny."
6. **Final Generation:** The agent synthesizes all observations into a distinct, accurate response natively avoiding hallucinations.

### Citations and Provenance

Because we use `Document` objects, the chunk returned by the retriever still has its `metadata` attached (like the source PDF name and page number). We can engineer our prompt to force the LLM to cite exactly which document it used to generate its answer, providing absolute provenance for enterprise audits.

> **Key Term - Provenance / Citation:** In AI systems, provenance means tracking the exact source document, page, and version that was used to generate each statement in a response. Provenance is critical for enterprise use cases in law, medicine, and finance, where claims must be auditable and traceable to authoritative sources.

## Code Example

```python
from langchain.chat_models import init_chat_model
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
# Assuming `vectorstore` was created previously as seen in Day 4

# 1. Create the Retriever Tool from the Vectorstore
retriever = vectorstore.as_retriever()

# We wrap the retriever in a Tool interface with a highly specific description
retriever_tool = create_retriever_tool(
    retriever,
    name="company_handbook_search",
    description="Search for information about company policies, PTO, and benefits. You must use this for any employee-related questions."
)

# 2. Setup the Agent Configuration
tools = [retriever_tool]
llm = init_chat_model("gpt-4o", model_provider="openai", temperature=0)

system_prompt = """You are an HR Assistant. Use the company_handbook_search tool to answer policy questions. 
If you use the tool, you MUST cite your source at the end of your response."""

# 3. Compile the Agentic RAG Graph
agent_executor = create_react_agent(llm, tools, state_modifier=system_prompt)

# 4. End-to-End Execution
# stream_mode="values" means we receive the full agent state after every step,
# not just the final message. This lets us print each reasoning/tool step as it happens.
events = agent_executor.stream(
    {"messages": [("user", "Can I roll my PTO over to next year?")]},
    stream_mode="values"  # Alternative: stream_mode="updates" for delta-only output
)

# Iterate through the execution loop to see the agent reasoning live
for event in events:
    event["messages"][-1].pretty_print()
```

## Additional Resources

- [Conversational RAG Agents Tutorial](https://python.langchain.com/docs/tutorials/qa_chat_history/)
- [LangGraph Agent with Retriever Tool](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)
