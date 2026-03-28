# Demo: LangGraph Cloud Deployment

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
| --- | --- |
| **Blocking vs. Non-Blocking** | *"Your FastAPI server has 4 worker threads. 5 users send requests to your LangGraph agent at the same time. Each request takes 30 seconds. Walk me through what happens to each user's request second-by-second WITHOUT async."* |
| **`async`/`await` Mechanics** | *"When Python hits `await graph.ainvoke(...)`, does execution stop entirely? What exactly does the Event Loop do while the network call is in progress? What's free to run during that time?"* |
| **Streaming vs. Full Response** | *"From the user's perspective, what is the VISIBLE difference between `graph.invoke()` and `graph.astream_events()`? Why does streaming dramatically reduce UI abandonment rates even when total completion time is identical?"* |
| **LangGraph Cloud** | *"What three things does LangGraph Cloud automatically provision that you'd have to build yourself if deploying as a plain Python script? Which of those three would be hardest to build correctly?"* |


## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/async-streaming-operations.mermaid`.
2. Trace the path from the React Frontend to the FastAPI server. Ask the class: "If you have a Python web server with 4 worker threads, and 5 users ask a complex Agent a question at the same time, what happens to User #5?" (Answer: Without async/await their request is blocked and the server times out. `ainvoke` solves this by yielding the thread while the LLM "thinks").
3. Walk through the events bridging the backend to the User Experience. Explain that `astream_events` opens a Web Socket (or Server-Sent Events). The server literally sends partial JSON chunks thousands of times before the graph completely finishes.

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d067-async-operations-and-streaming.py`.
2. Notice the two functions at the bottom: `simulate_synchronous_blocking_request` vs `simulate_asynchronous_streaming_request`.
3. In `build_async_graph`, point out that our node logic uses `async def` and `await asyncio.sleep(...)`. 
4. Execute the script. 
5. During the Synchronous invocation, point out the dead, frozen 4-second silence in the terminal. The user hates this.
6. During the Streaming invocation, watch the terminal timestamps.
   - At `2.5s`, it prints the Database completion.
   - At `2.6s`, it starts typewriter animating the text dynamically using token word chunks. 
   - This prevents user abandonment because the UX proves the system is working instantly.

## Summary
Reiterate that mastering Async/Await and Event Streams separates hobbyist local data scientists from production software engineers deploying scalable AI infra to the cloud.
