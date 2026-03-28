# Demo: Runtime Configuration and Middleware

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
| --- | --- |
| **Runtime Configuration** | *"If you hardcode `model_name = 'gpt-4'` in your graph code, what's the deployment process every time you want to A/B test a different model? How does runtime `configurable` solve this?"* |
| **Message Trimming** | *"If you never trim the conversation history, a user who chats for 2 hours would eventually break the application. At what EXACT point does the crash happen, and what error would you see?"* |
| **Middleware (Token Economics)** | *"GPT-4 costs ~$0.03 per 1000 tokens. A 500-message conversation history might be 50,000 tokens. How much would a single API call cost? Why is this unsustainable at scale?"* |
| **Security via State Injection** | *"Why is it safer to inject a `user_id` into a delete tool via the graph's runtime context rather than letting the LLM pass it as an argument? What attack is this defending against?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/toolruntime-state.mermaid`.
2. Trace the path emphasizing the two isolated problems being solved.
3. **Problem 1: Token Limits.** Ask the class: "If a user chats with your bot 500 times in one day, what happens if you pass the entire `messages` array to GPT-4?" (Answer: The API will crash with a Context Window Exceeded error, and it will cost $5 to process one message. We MUST trim history). Point out the Middleware block that slices the array but *preserves* the System Prompt.
4. **Problem 2: Security/State.** Point to the `Graph Config` injecting variables into the `Tool Node`. Ask: "If a user wants to delete their account, why is it dangerous to let the LLM generate the `user_id` argument for the `delete_account()` tool?" (Answer: The LLM could hallucinate or be prompt-injected to pass `user_id="admin"` and delete the CEO's account. Secure variables must bypass the LLM entirely via context injection).

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d061-toolruntime-and-state-management.py`.
2. Run the script via `python d061-toolruntime-and-state-management.py` so the terminal output is visible as you walk through.
3. Review `simulate_message_trimming()`. Show the dummy history. Emphasize that throwing away the "weather" messages makes the LLM faster and cheaper while retaining exactly what it needs to execute the current task. **Highlight the rule: Never trim index 0 (the System Prompt)!**
4. Review `simulate_tool_state_injection()`. 
5. Explain that the `graph_runnable_config` represents variables injected by the web server (like FastAPI) at the moment of API invocation based on the user's JWT token. 
6. Show how `llm_generated_args` only contains `amount` and `recipient`, but the tool magically executes with `injected_user_id="VALID_USER_777"`. This separation of concerns is vital for enterprise architecture.

## Summary
Summarize that scaling agents from prototypes to production requires brutal management of both Tokens (via Trimming) and Security/Context (via Runtime Injection). Relying on the LLM to manage all state natively is an architectural anti-pattern.
