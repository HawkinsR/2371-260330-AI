# Lab: Conditional Routing Graph

## The Scenario
Your support ticketing system is overwhelmed. To ensure your "Gold" tier enterprise customers receive immediate attention, you must design a LangGraph architecture that conditionally routes tickets based on the user's loyalty status. You will build a system with a central `AgentState`, a categorizing node that looks up the user's tier, and two distinct support nodes. Crucially, you will write the custom router function to tie them together dynamically.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e046-e046-routing_graph_lab.py`.
3. Complete the `AgentState` schema:
   - Define a `TypedDict` containing three keys: `user_input` (str), `loyalty_tier` (str), and `final_answer` (str).
4. Complete the `route_by_tier` function:
   - This represents the conditional edge. Look at `state.get("loyalty_tier")`.
   - If the tier is `"gold"`, return the exact string `"priority"`.
   - Otherwise, return `"standard"`.
5. Complete the graph assembly in `build_support_graph`:
   - Add the three nodes: `"analyzer"` maps to `analyze_tier_node`, `"priority"` maps to `priority_support_node`, and `"standard"` maps to `standard_support_node`.
   - Add a static edge from `START` to `"analyzer"`.
   - Add the conditional edges originating from `"analyzer"`, utilizing your `route_by_tier` function.
   - Map the router strings to the node names in the condition dictionary: `{"priority": "priority", "standard": "standard"}`.
   - Add static edges from `"priority"` to `END`, and `"standard"` to `END`.
   - Compile the graph.

## Definition of Done
- The script executes successfully and compiles a LangGraph application without errors.
- Scenario A outputs a "standard" support response for the silver-tier customer.
- Scenario B outputs a "priority" escalated response for the gold-tier enterprise customer.
