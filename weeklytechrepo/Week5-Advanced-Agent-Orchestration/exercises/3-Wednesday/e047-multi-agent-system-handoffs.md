# E047: Multi-Agent System Handoffs

## Objective
Build a multi-agent system Orchestrator/Worker pattern. A central Bedrock orchestrator will evaluate user intent, identify if research is necessary, hand off the context to a search worker, and then aggregate the final result.

## Instructions
1. Open `e047-multi-agent.py` inside the `starter_code/` directory.
2. Initialize an Orchestrator LLM using `ChatBedrock`.
3. Build two separate node handlers: `orchestrator_node` and `research_worker_node`. 
4. The Orchestrator must read the user query. If the query requires finding live information, it issues a `Command(goto="research_worker")`. If it does not, it commands a jump to the `END` state.
5. Compile the main `StateGraph` linking the supervisor and the sub-agent.
6. Test it with two distinct prompts to prove the orchestrator intelligently delegates only when necessary.
