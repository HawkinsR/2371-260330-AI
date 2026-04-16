# LangChain Foundations and Agents

## Learning Objectives

- Build a Basic LangChain Agent using the `create_react_agent` logic.
- Configure **Amazon Bedrock** models using the `init_chat_model` Universal Interface.
- Manipulate Chat Models behaviorally using System Prompts and Structured Output (Pydantic).
- Implement Streaming Responses to handle real-time UI/UX requirements.
- Understand Token Economics and the cost implications of agentic loops in a cloud environment.

## Why This Matters

Language models are impressive, but they are trapped inside a chat window; they cannot *do* anything. Agentic AI bridges this gap. By giving an LLM access to external tools (like a calculator, database, or API) and teaching it the "Reason + Act" (ReAct) paradigm, we transform a passive chatbot into an active agent capable of autonomously solving multi-step problems. Using **Amazon Bedrock** ensures these agents run on enterprise-grade infrastructure with robust security and compliance.

> **Key Term - Agentic AI:** An AI system that goes beyond generating text to autonomously take actions in the world. An agent perceives its environment, reasons about what to do, uses tools (like web search, code execution, or database queries) to gather information or perform tasks, and iterates until it achieves a goal — without requiring a human prompt at every step.

## The Concept

### The ReAct Paradigm and Tools

"ReAct" stands for **Reason** and **Act**. Instead of just answering a prompt, the agent:

1. **Reasons** about what the user wants.
2. **Acts** by selecting a specific tool to gather data or perform a task.
3. **Observes** the output of that tool.
4. Repeats the loop until it has enough context to formulate a final answer.

To enable this, we use the `@tool` decorator in LangChain to wrap standard Python functions. The LLM reads the function's docstring and type hints to understand *when* and *how* to call it.

> **Key Term - ReAct Paradigm:** A prompting and execution strategy for AI agents. The agent alternates between **Re**asoning (thinking about what it needs) and **Act**ing (calling a tool). Unlike a single-shot LLM call, ReAct loops allow the agent to gather information incrementally.

### Bedrock Inference Parameters

When using Amazon Bedrock, we must manage **Inference Parameters** to control the model's creativity and deterministic nature:
- **Temperature:** Controls randomness (0.0 for deterministic tool-use).
- **Top P (Nucleus Sampling):** Filters the cumulative probability of next tokens.
- **Max Tokens:** The hard limit on the response length.

### System Prompts and Structured Output

LLMs are probabilistic. To enforce professional behavior, we provide a **System Prompt**—a set of inviolable rules established before the user ever sends a message. Furthermore, when integrating LLMs into software pipelines, we force the LLM to return strictly formatted data (like JSON) using **Pydantic** schemas.

### Streaming and Token Economics

For production-grade agents, waiting 10 seconds for a full answer is unacceptable UI/UX. **Streaming** allows the agent to send tokens to the client as they are generated. Tracking **Token Economics** on Bedrock involves monitoring "Input Tokens" (the prompt + history) and "Output Tokens" (the model's generated text), which are billed per 1,000 tokens.

> **Key Term - Token Economics:** The study of managing the cost and performance of LLM applications. Optimization involves choosing the right model size (e.g., Claude 3.5 Sonnet for reasoning vs. Haiku for simple tasks), pruning history, and using efficient system prompts.

## Code Example

```python
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# 1. Initialize the Chat Model interface via Amazon Bedrock
# We use Claude 3.5 Sonnet for high-reasoning tasks
llm = init_chat_model(
    model="us.anthropic.claude-3-5-sonnet-20240620-v1:0", 
    model_provider="bedrock", 
    temperature=0
)

# 2. Define a Tool
@tool
def get_stock_price(ticker: str) -> str:
    """Returns the current stock price for a given ticker symbol."""
    # Simulation of a real financial API call
    data = {"AAPL": "$230.15", "AMZN": "$185.40", "GOOGL": "$172.10"}
    return data.get(ticker.upper(), "Ticker not found.")

# 3. Create the ReAct Agent Graph
system_prompt = "You are a financial analyst. Use the stock tool for any price queries."
tools = [get_stock_price]
agent_executor = create_react_agent(llm, tools, state_modifier=system_prompt)

# 4. Invoke the Agent (Streaming results)
query = {"messages": [("user", "What is the current price of Apple stock?")]}
for chunk in agent_executor.stream(query, stream_mode="values"):
    message = chunk["messages"][-1]
    print(f"[{message.type.upper()}]: {message.content}")
```

## Additional Resources

- [LangChain Bedrock Integration](https://python.langchain.com/docs/integrations/chat/bedrock/)
- [Amazon Bedrock Inference Parameters](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html)
- [LangGraph Prebuilt Agents](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/)
