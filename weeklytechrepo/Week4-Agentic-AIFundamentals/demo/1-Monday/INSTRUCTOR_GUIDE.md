# Instructor Guide: LangChain Bedrock Foundations

## Overview
This demo introduces the **Reason + Act (ReAct)** paradigm using **Amazon Bedrock**. We focus on building a robust agent that can use tools while maintaining a strict output format (Pydantic) and providing a modern streaming experience.

## Phase 1: The Concept (Whiteboard)
**Time:** 10 mins

1.  **Open `diagrams/bedrock_react.mermaid`**.
2.  **The Loop**: Show how the "Reason" phase (LLM thinking) leads to an "Act" phase (Tool call).
3.  **Bedrock Role**: Explain that Bedrock is the *runtime*. It provides the "brain" via models like Claude 3.5 Sonnet.
4.  **Discussion**: Ask the trainees: "Why does the LLM need a tool instead of just answering directly?" (Answer: To access real-time or private data it wasn't trained on).

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins

1.  **Open `code/d054-langchain-bedrock-foundations.py`**.
2.  **Initialization**:
    - Highlight `init_chat_model` with `model_provider="bedrock"`. 
    - Explain **Inference Parameters** (Temperature=0 for deterministic tool use).
3.  **Tool Definition**:
    - Show the `@tool` decorator. 
    - Point out why docstrings are critical (the LLM reads them to know *how* to use the tool).
4.  **The Agent**:
    - Explain `create_react_agent`. Explain that this is a "helper" that constructs a LangGraph state machine for us.
5.  **Streaming**:
    - Demonstrate the `.stream()` method. Explain that we use `stream_mode="values"` to see the full history update.

## Summary Checklist for Trainees
- [ ] Is my AWS region set correctly?
- [ ] Have I requested access to Claude 3.5 Sonnet in the Bedrock console?
- [ ] Does my tool have a clear docstring and type hints?
