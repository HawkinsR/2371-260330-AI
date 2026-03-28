# Agentic Design and ReAct

## Learning Objectives

- Define Agentic Design Patterns and their role in autonomous GenAI loops.
- Configure universally compatible LLMs using the `init_chat_model` Universal Interface.
- Manipulate Chat Models behaviorally using Temperature and System Prompts vs User Prompts.
- Construct executable Python functions and expose them using The `@tool` Decorator.
- Enforce deterministic JSON output using Structured Output with Pydantic.
- Compile a Simple ReAct Agent Creation loop (`create_agent`) to handle multi-step reasoning.

## Why This Matters

Language models like ChatGPT are impressive, but they are trapped inside a chat window; they cannot *do* anything. Agentic AI bridges this gap. By giving an LLM access to external tools (like a calculator, database, or API) and teaching it the "Reason + Act" (ReAct) paradigm, we transform a passive chatbot into an active agent capable of autonomously solving multi-step problems.

> **Key Term - Agentic AI:** An AI system that goes beyond generating text to autonomously take actions in the world. An agent perceives its environment, reasons about what to do, uses tools (like web search, code execution, or database queries) to gather information or perform tasks, and iterates until it achieves a goal — without requiring a human prompt at every step.

## The Concept

### The ReAct Paradigm and Tools

"ReAct" stands for **Reason** and **Act**. Instead of just answering a prompt, the agent:

1. **Reasons** about what the user wants.
2. **Acts** by selecting a specific tool to gather data or perform a task.
3. **Observes** the output of that tool.
4. Repeats the loop until it has enough context to formulate a final answer.
To enable this, we use the `@tool` decorator in LangChain to wrap standard Python functions. The LLM reads the function's docstring and type hints to understand *when* and *how* to call it.

> **Key Term - ReAct Paradigm:** A prompting and execution strategy for AI agents. The agent alternates between **Re**asoning (thinking about what it needs) and **Act**ing (calling a tool). Unlike a single-shot LLM call, ReAct loops allow the agent to gather information incrementally — checking its own work and course-correcting across multiple steps.

> **Key Term - `@tool` Decorator:** A LangChain annotation that transforms a normal Python function into an LLM-callable tool. The LLM reads the function's name, docstring, and type hints to understand what the tool does and what arguments it requires — automatically deciding when to call it based on the user's intent.

### System Prompts and Structured Output

LLMs are probabilistic. To enforce professional behavior, we provide a **System Prompt**—a set of inviolable rules established before the user ever sends a message. Furthermore, when integrating LLMs into software pipelines, we cannot accept chaotic text strings. We force the LLM to return strictly formatted data (like JSON) using `Pydantic` schemas, guaranteeing the output is parsable by the rest of our application.

> **Key Term - System Prompt:** Hidden instructions sent to an LLM before the user's message. System prompts establish the AI's persona, rules, and constraints (e.g., "You are a professional financial advisor. Never give specific stock advice."). Unlike user prompts, system prompts persist across the entire conversation and are not visible to the end user.

> **Key Term - Pydantic:** A Python library for data validation. In AI applications, Pydantic models define the exact JSON structure that an LLM must output (e.g., `{"city": str, "temperature": int, "unit": Literal["C", "F"]}`). If the LLM's output doesn't match the schema, Pydantic raises an error — preventing malformed data from propagating downstream.

> **Key Term - Probabilistic Output:** LLMs do not always produce identical output for the same input — the `temperature` parameter controls this randomness. A temperature of `0` makes the model deterministic (always picks the most likely next token). A temperature of `1.0` introduces creative variability. For production pipelines and tool calling, `temperature=0` is the standard choice.

## Code Example

```python
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
# Assuming API keys are set in the environment

# 1. Initialize the Chat Model interface
llm = init_chat_model("gpt-4o", model_provider="openai", temperature=0)

# 2. Define a Tool
@tool
def get_weather(location: str) -> str:
    """Returns the current weather in a given city."""
    # In reality, this would call an external API
    if location.lower() == "miami":
        return "It is currently 85 degrees and sunny in Miami."
    return "Weather data unavailable."

# 3. Create the ReAct Agent Graph
system_prompt = "You are a helpful assistant. Always use tools to verify facts."
tools = [get_weather]
agent_executor = create_react_agent(llm, tools, state_modifier=system_prompt)

# 4. Invoke the Agent (It will automatically decide to call the weather tool)
response = agent_executor.invoke({"messages": [("user", "Should I pack a coat for Miami?")]})
print(response["messages"][-1].content)
```

## Additional Resources

- [LangChain Chat Model Universal Init](https://python.langchain.com/docs/how_to/chat_models_universal_init/)
- [LangGraph Prebuilt Agents](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/)
