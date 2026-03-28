# Context Engineering and LangSmith

## Learning Objectives

- Design flexible prompt templates through Context Engineering Implementation.
- Dynamically inject variables into System Prompts utilizing LangChain structures.
- Navigate the LangChain Standard Application Structure integrating formatters and parsers.
- Debug agent workflows structurally by Tracing Agent Execution.
- Analyze token latency and cost metrics configuring LangSmith.
- Create automated Datasets in LangSmith for deterministic evaluation workflows.

## Why This Matters

As agent workflows grow from single LLM calls into multi-tool reasoning loops, "Print statement debugging" totally fails. You need a dedicated, graphical orchestration dashboard to track exactly what the AI was thinking at each step, how long the external API call took, and how many tokens were consumed. LangSmith provides this observability, turning chaotic LLM calls into deterministic, optimized software pipelines heavily driven by engineered context windows.

> **Key Term - Context Window:** The maximum amount of text (measured in "tokens") that an LLM can read and process in a single request. GPT-4 supports ~128,000 tokens (roughly 100,000 words). If your input exceeds this limit, the LLM cannot process it. Context window management — deciding what to include and cut — is a primary engineering challenge in agentic systems.

> **Key Term - Token:** The basic unit of text that LLMs process. Tokens roughly correspond to word fragments — for example, "unhappiness" may be split into ["un", "happi", "ness"] = 3 tokens. Both input text and output text consume tokens, and LLM API costs are billed per token.

## The Concept

### Dynamic System Prompts and Context Engineering

Static strings are rarely useful in production. We use Prompt Templates to dynamically inject variables (like the current user's name, or the output of a prior database query) directly into the System Prompt just before the LLM fires. This is the essence of Context Engineering: perfectly sculpting the text window to give the LLM exactly the context it needs, without wasting expensive tokens on irrelevant information.

> **Key Term - Context Engineering:** The practice of strategically controlling what text occupies the LLM's context window. Unlike prompt engineering (which focuses on the instructions), context engineering manages the surrounding data — choosing which documents, memories, or tool outputs to include, and ensuring the total stays under the token limit.

### Tracing Execution with LangSmith

LangSmith is the observability platform built specifically for LangChain/LangGraph. When you enable tracing (simply by setting environment variables), every single step of your agent's execution is logged to a secure cloud dashboard. You can visually inspect the tree of logic:

- How long did the HTTP request take?
- Exactly what raw payload was sent to OpenAI?
- Why did the agent choose Tool A instead of Tool B?

This tracing allows you to pinpoint issues quickly. When reading a trace in LangSmith, focus on three things: the **Input** (what prompt the model actually received), the **Output** (what it returned), and the **Latency/Token Count** column (which identifies expensive or slow steps to optimize).

> **Key Term - Hallucination:** When an LLM confidently generates text that sounds plausible but is factually incorrect. For example, an LLM might invent a fake citation or describe a policy that doesn't exist. Hallucinations occur because LLMs generate the statistically likely next word, not the true next word. RAG and citations are architectural strategies to reduce hallucinations by grounding responses in retrieved facts.

> **Key Term - Observability:** A software engineering principle measuring how well you can understand the internal state of a system by examining its outputs. For LLM applications, observability means being able to answer: "What prompt did the model receive? Which tool did it call? Why did it choose that response?" LangSmith provides this observability layer for AI workflows.

### Datasets and Handling Window Limits

You can convert successful trace runs into "Golden Datasets" within LangSmith. These datasets act as Unit Tests for AI. Before pushing an updated prompt to production, you run the new prompt against the dataset to ensure it hasn't regressed on handling context window limits or formatting requirements.

## Code Example

```python
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

# Enable LangSmith Tracing via Environment Variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Trainee-Context-Project"
# os.environ["LANGCHAIN_API_KEY"] = "your-key-here"

# 1. Context Engineering: Dynamic Prompt Templates
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert on {topic}. Answer strictly in {language}."),
    ("user", "{question}")
])

# 2. Standard Application Structure (Chaining)
llm = init_chat_model("gpt-3.5-turbo", model_provider="openai")

# The | operator chains the prompt formatting directly into the LLM call
processing_chain = prompt_template | llm

# 3. Execution (This run is silently traced and sent to LangSmith)
result = processing_chain.invoke({
    "topic": "Astrophysics",
    "language": "French",
    "question": "What is a black hole?"
})

print(result.content)
# Log into smith.langchain.com to view the exact token cost and latency of this run!
```

## Additional Resources

- [LangSmith Tracing Documentation](https://docs.smith.langchain.com/observability)
- [LangChain Prompt Templates](https://python.langchain.com/docs/concepts/prompt_templates/)
