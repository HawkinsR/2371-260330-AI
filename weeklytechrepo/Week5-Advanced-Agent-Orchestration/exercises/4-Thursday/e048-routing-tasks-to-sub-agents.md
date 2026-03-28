# Lab: Supervisor and Worker Agents Architecture

## The Scenario
Your content creation team is struggling to produce high-quality articles. When a single LLM tries to research a topic *and* write the final article simultaneously, the results are generic and often hallucinated. You have been tasked with building a Multi-Agent system. You must implement a "Supervisor" node that evaluates the global state and dynamically routes tasks to either the `Researcher` agent or the `Editor` agent until the article is complete.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e048-e048-supervisor_lab.py`.
3. Complete the `researcher_agent_node` function:
   - This function takes the global `state`.
   - Update the state by returning a dictionary with `{"research_notes": "Found 3 verified sources regarding AI enterprise adoption."}`
4. Complete the `editor_agent_node` function:
   - This function takes the global `state`.
   - Update the state by returning a dictionary with `{"final_article": "AI is transforming enterprise software development."}`
5. Complete the `supervisor_node` function:
   - This function acts as the orchestrator logic. It evaluates the `state` dictionary.
   - If `state.get("research_notes")` is missing or empty, return `{"next_action": "Researcher"}`.
   - If `state.get("research_notes")` exists but `state.get("final_article")` is missing or empty, return `{"next_action": "Editor"}`.
   - If both exist, return `{"next_action": "FINISH"}`.

## Definition of Done
- The script executes successfully without entering an infinite loop.
- The supervisor correctly routes the state to the Researcher first, then the Editor, and finally triggers FINISH.
- The console outputs the final article and research notes upon pipeline completion.
