# Demo: Agentic Design and ReAct

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
| --- | --- |
| **Agentic AI** | *"What can ChatGPT NOT do on its own? Give an example of something a human would still need to do manually after getting a ChatGPT response. How would an agent close that gap?"* |
| **ReAct Loop** | *"A chess player doesn't plan their entire game before move 1. They make a move, observe the board, re-evaluate, then make the next move. How does that describe the ReAct pattern?"* |
| **`@tool` Decorator** | *"When you write a Python function docstring, who normally reads it? How does it change the picture when the reader is an LLM deciding whether to call your function?"* |
| **Structured Output / Pydantic** | *"If your React frontend calls an API and sometimes gets back JSON and sometimes gets back a paragraph of English text, what breaks? Why is Pydantic so critical in LLM pipelines?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/react-agent-flow.mermaid`.
2. Trace the loop. Ask the class: "What happens if a standard ChatGPT model without tools is asked for today's weather in Miami?" (Answer: It either hallucinates or says 'I am an AI and cannot access real-time data').
3. Explain the **Observation** step. The Python function literally executes, retrieves string/json data, and appends it to the LLM's conversation history as a "ToolMessage". The LLM reads it, and *Re-Reasons* what to do next.
4. Point to the **Structured Output** box. Emphasize that passing unpredictable paragraphs of text between software components breaks Python applications. We use Pydantic to force the LLM to output predictable JSON models.

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d054-simple-react-agent.py`.
2. Review the two mock tool functions. Notice their clear docstrings and type hints. LangChain automatically reads these type hints to teach the LLM exactly how to call the function natively.
3. Review `PortfolioReport(TypedDict)`. This represents our Pydantic schema enforcing structured data.
4. Execute the script via `run_demo()`. 
5. Walk through the terminal output line by line. Emphasize the `AGENT THOUGHT` -> `Tool Execution` -> `AGENT OBSERVATION` cadence. This is the ReAct loop unfolding sequentially.
6. Look at the `FINAL APPLICATION PAYLOAD`. Reveal that the output is perfect JSON. A frontend React developer can easily digest `payload.total_value` to render a dashboard widget effortlessly.

## Summary
Reiterate that LLMs are powerful reasoning engines. By combining them with Python Tools and structured Pydantic outputs, we transform text-generators into autonomous software operators.
