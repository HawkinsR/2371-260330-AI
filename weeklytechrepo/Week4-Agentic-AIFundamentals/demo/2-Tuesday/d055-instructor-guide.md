# Demo: Context Engineering and LangSmith

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
| --- | --- |
| **Context Window / Tokens** | *"If the user's conversation history grows to 200,000 words but GPT-4 only supports ~100,000 words, what happens to the beginning of the conversation? What strategy would you use to handle this?"* |
| **Context Engineering** | *"Why isn't just writing a longer system prompt always the answer? What are the costs and tradeoffs of including more text in every single LLM call?"* |
| **Hallucination** | *"An LLM confidently tells a user they qualify for a policy benefit that doesn't exist. How does that happen technically? What architectural pattern prevents it?"* |
| **Observability / LangSmith** | *"If a 10-step agent workflow returns the wrong answer, print statements tell you the final output was wrong. What else would you need to pinpoint exactly which step caused the failure?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/dynamic-prompts-tracing.mermaid`.
2. Trace the path emphasizing the difference between the **User Prompt** and the **System Message**. Ask the class: "Why don't we just append our instructions to the user's prompt string directly?" (Answer: Security and structure. The System Message explicitly tells the LLM the rules of the environment and operates independently of user manipulation. Injecting variables like Database metrics into the System prompt ensures safe Context Engineering).
3. Focus on the `LangSmith Telemetry (Invisible)` box. Explain that tracing doesn't require complex API coding; the LangChain SDK natively broadcasts execution telemetry directly to the cloud if the environment variables are active. Latency, exact prompts, and costs are tracked effortlessly.

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d055-dynamic-prompts-and-tracing.py`.
2. Walk through `configure_langsmith_tracing()`. Show how simple the configuration is `os.environ["LANGCHAIN_TRACING_V2"] = "true"`.
3. Walk through the `system_template` in `compile_dynamic_prompt()`. Point out the `{name}` and `{tier}` placeholders. 
4. Explain that this is the core of AI software engineering: transforming business logic (is the user "Free" or "Enterprise") from database booleans into plain-text logical constraints the LLM can process.
5. Execute the script via `run_context_demo()`. 
6. Show the result. Because the `user_tier` was set to "Free", the LLM natively appended the upsell sentence exactly as the System Rule dictated.
7. Finally, point out the `[LANGSMITH TRACE OVERVIEW]`. Emphasize that in production, tracking metric anomalies like a sudden spike to `Latency: 45.0s` or mapping high `Total Cost` directly back to the `user_query` is what separates hobbyists from enterprise engineers.

## Summary
Reiterate that powerful prompt engineering is *not* typing magic paragraphs into ChatGPT. In software development, Context Engineering means dynamically assembling strings based on external SQL lookups and strict template injection, entirely governed by deep-layer tracing metrics in platforms like LangSmith.
