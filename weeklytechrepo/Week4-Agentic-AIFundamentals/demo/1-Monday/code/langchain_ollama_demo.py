from langchain_ollama import ChatOllama
from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"It's always sunny in {city}"

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
).bind_tools([get_weather])

response = llm.invoke(
    "What is the weather in Chicago?"
)

print(response.content)
print(response.tool_calls)
