# Demo: Self-Correction and Plan and Execute

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
| --- | --- |
| **Zero-Shot Failure** | *"You ask GPT-4 to write a full legal contract in a single prompt. What are three specific ways the output is likely to be wrong or incomplete? How does adding a Reflection loop address each one?"* |
| **Generator vs. Evaluator Roles** | *"In a self-correction loop, could a SINGLE LLM node both generate the draft AND evaluate it in the same call? What's the architectural risk of that approach compared to using two separate nodes?"* |
| **Plan-and-Execute** | *"Why does the Planner use structured JSON output (a step-by-step checklist) instead of just describing the plan in plain English? What breaks in the Executor node if the plan is unstructured?"* |
| **Loop Guard** | *"If the Evaluator always returns 'Needs Improvement' due to an overly strict prompt, what happens without a `revision_count >= 3` guard? What real-world cost does this represent?"* |


## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/iterative-refinement-flow.mermaid`.
2. Trace the path from the User Query to the `Evaluator`. Explain that in standard zero-shot LLM queries, whatever the `Generator` outputted first goes straight to the user.
3. Walk through the **Iterative Refinement Loop** subgraph. 
4. **Discussion:** Point to the **Failsafe** section. Ask the class: "Why must we hardcode a 'Loop Limit Reached' exit?" (Answer: Without an explicit max iteration count in the `should_continue` routing logic, a particularly stubborn Evaluator could trap the Generator in an infinite loop, endlessly draining API budget).

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d065-iterative-refinement-agent.py`.
2. Review the `Evaluator Node`. 
   - Point out the strict rules (length > 15, must contain "persistent state"). Explain that in production, this node is usually an LLM Prompt tasked explicitly with finding flaws, rather than hardcoded Python strings. Let the LLM critique the LLM.
3. Review the `should_continue` routing logic. Show how it parses the `is_perfect` boolean returned by the Evaluator, mapping `"continue"` backward to the Generator and `"end"` to the `END` boundary.
4. Execute the script via `demonstrate_self_correction()`. 
5. Walk the class through the verbose terminal trace. Watch how the Initial Draft is generated, fails Rule 1, gets rewritten, fails Rule 2, gets rewritten again, and finally passes.

## Summary
Reiterate that separating the roles of "Generator" and "Editor" dramatically reduces hallucinations and format errors, converting erratic AI outputs into reliable engineering pipelines.
