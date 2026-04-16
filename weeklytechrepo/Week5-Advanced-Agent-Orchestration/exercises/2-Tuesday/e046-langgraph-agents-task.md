# E046: Building LangGraph Agents

## Objective
Construct a functional `StateGraph` that binds a basic core Python tool to an AWS `ChatBedrock` model. The LLM must be granted the capability to trigger the tool when necessary. 

## Instructions
1. Open the starter code at `starter_code/e046-langgraph-agents.py`.
2. Define a simple mathematical tool using LangChain's `@tool` decorator (e.g., `def multiply(a: int, b: int) -> int:`).
3. Initialize a `ChatBedrock` model. Use `.bind_tools()` to make the model aware of your math tool.
4. Set up an `AgentState` TypedDict to manage `messages`.
5. Create a `StateGraph`. Add your LLM agent node, and add the `ToolNode` (from `langgraph.prebuilt`) to handle tool executions.
6. Configure conditional edges so the graph checks if the LLM returned a tool call; if so, route to the tool. Otherwise, `END` execution.
7. Compile the graph and test it with both a standard greeting and a math question.
