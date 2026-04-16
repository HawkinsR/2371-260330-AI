# Exercise e040: Building Your First Bedrock Agent

## Overview
In this exercise, you will build a real LangChain agent backed by **Amazon Bedrock**. The agent will use the `@tool` decorator to expose a custom function, stream its responses token-by-token, and enforce structured output using **Pydantic**.

## Learning Outcomes
- Initialize a Bedrock-backed LLM with `init_chat_model`.
- Define a callable `@tool` with a proper docstring and type hints.
- Stream agent responses using `.stream()` and `stream_mode="values"`.
- Validate agent output with a Pydantic schema.

## Prerequisites
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION` set in your environment.
- Access to `anthropic.claude-3-5-sonnet` enabled in your Bedrock console.
- `pip install langchain-aws langgraph pydantic`

## Instructions

Open `starter_code/e040-bedrock-agent-lab.py`. You will complete the TODOs in order:

1. **Define the Pydantic Schema** — Create a `StockRecommendation` model with fields for `ticker`, `recommendation`, and `reasoning`.
2. **Define the `@tool`** — Write a `get_stock_sentiment(ticker: str) -> str` function that returns mock sentiment data for at least 3 tickers.
3. **Initialize Bedrock** — Use `init_chat_model` with `model_provider="bedrock"` and `temperature=0`.
4. **Create the Agent** — Use `create_react_agent` with your LLM, tool, and a professional system prompt.
5. **Stream the Invocation** — Use `.stream(query, stream_mode="values")` and print messages as they arrive.

## Deliverable
Run the agent against the query: `"What is your recommendation for Tesla (TSLA) stock?"` and observe the full ReAct loop streaming live to the terminal.
