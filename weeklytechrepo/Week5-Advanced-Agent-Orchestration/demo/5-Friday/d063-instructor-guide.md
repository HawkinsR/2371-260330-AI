# Demo: Breakpoints and Time Travel

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
| --- | --- |
| **Human-in-the-Loop** | *"Name 3 specific AI actions where fully autonomous execution without human approval would be unacceptable in an enterprise context. What's the financial, legal, or reputational risk in each case?"* |
| **Breakpoint / Interrupt** | *"When the graph hits an `interrupt_before` breakpoint and pauses, does the Python process crash? Does it wait forever? What actually happens in memory, and how does it persist?"* |
| **Checkpointer** | *"If your AI graph crashes at Node 8 of 12 due to a network error, and you restart without a Checkpointer, what happens? With a Checkpointer, what's possible?"* |
| **Time Travel / State Editing** | *"An agent ran 10 steps and made a wrong decision at Step 4. With Time Travel, can you fix Step 4 without redoing Steps 1-3? What makes this possible technically?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/graph-interrupts-editing.mermaid`.
2. Trace the path down to the `Interrupt Flag`. Ask the class: "If an agent has a tool to `execute_stock_trade(ticker, amount)`, why must we use an interrupt?" (Answer: Hallucinations are inevitable. Autonomous execution of financially or legally binding actions without a human circuit-breaker is negligent).
3. Walk through the Human-in-the-Loop decision tree. 
4. Explain "Time Travel". Because the Checkpointer saves a snapshot of the State dictionary at *every single node transition*, we can literally look at the "history" of the graph, grab the State at Step 2, manually alter variables like `amount=500` instead of `amount=50000`, and tell the graph to resume execution from Step 2 as if the hallucination never happened.

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d063-graph-interrupts-and-state-editing.py`.
2. Notice the imports: `MemorySaver`. This is the checkpointer DB enabling State persistence across physical time.
3. Review `build_interruptible_graph()`. Point out `interrupt_before=["SendEmail", "Cancel"]`. The graph physically suspends the python process right before these execute.
4. Execute the script via `run_human_in_the_loop_demo()`.
5. Point out the `!!! GRAPH EXECUTION PAUSED AT BREAKPOINT !!!` logs.
6. Look at Phase 3 in the code. We use `graph.update_state(...)` to manually overwrite the AI's bad 90% discount. We inject `is_approved: False`.
7. Look at Phase 4. We call `graph.invoke(None, config=config)`. Emphasize the `None`. We provide no new input; the graph just wakes up, looks at the DB, realizes the state now says `is_approved: False`, and cleanly routes to the `Cancel` node instead of sending the bad email.

## Summary
Reiterate that Breakpoints and State Overrides are the control panels of AI applications. They transform reckless autonomous scripts into heavily governed, enterprise-grade copilots that keep the human explicitly in the driver's seat for critical decisions.
