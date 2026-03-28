# Lab: Async Operations and Streaming

## The Scenario
Your company has built a fantastic "Data Analysis Agent." However, when you deployed it to your web server using standard `invoke()`, users reported that the web page would completely freeze for 10 seconds while the agent processed their request, leading them to refresh the page and break the application. You must upgrade the agent's architecture to use asynchronous python (`async`/`await`) and implement LangGraph's streaming events API (`astream_events`) so the frontend can receive live progress updates.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e053-e053-async_streaming_lab.py`.
3. Complete the `analysis_node` function:
   - This node takes 2 seconds to run. You must yield control back to the event loop so the server doesn't freeze.
   - Add the `await` keyword before `asyncio.sleep(2.0)`.
4. Complete the `run_streaming_agent` function:
   - Open a persistent streaming connection to the compiled `graph`.
   - Call `graph.astream_events(...)` passing `{"query": "Analyze Q3 revenue."}` as the input, and setting `version="v2"`. Store this generator in a variable named `events`.
5. Complete the asynchronous loop:
   - Iterate through the generator using `async for event in events:`.
   - Extract the event kind using `kind = event["event"]`.
   - Extract the node name using `node_name = event["name"]`.
   - Write an `if` statement: If `kind` equals `"on_chain_end"` AND `node_name` equals `"DataAnalyzer"`, print a success message to the console simulating a frontend UI update (e.g., `print("  [UI] Data Analysis Complete!")`).

## Definition of Done
- The script executes successfully without freezing.
- The console outputs the starting message, waits 2 seconds, and then outputs the `[UI] Data Analysis Complete!` message synchronously as the event stream fires.
