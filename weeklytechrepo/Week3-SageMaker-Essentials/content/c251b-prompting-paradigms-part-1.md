# Prompting Paradigms Part 1

## Learning Objectives

- Apply Zero-Shot and Few-Shot Prompting to guide model behavior.
- Implement Chain of Thought (CoT) to improve reasoning for complex tasks.
- Explore Tree of Thought (ToT) and Graph of Thought (GoT) for non-linear problem solving.
- Design curated Dialog State and Graph of Thought structures for multi-step interactions.

## Why This Matters

Early interaction with LLMs involved simple questions and answers. However, as task complexity increases, "linear" prompting often fails. Modern engineering requires structured prompting paradigms that force the model to reason, backtrack, and synthesize information. Understanding these architectures allows you to build systems that can solve logic puzzles, write complex code, and maintain state over long conversations — tasks that a simple "zero-shot" prompt cannot handle reliably.

This is not purely theoretical: in the context of this week's epic, every model you train and deploy on SageMaker will ultimately be *invoked through a prompt*. How well you structure that prompt is what separates a generic model response from a precision-engineered production output.

## The Concept

### Zero-Shot and Few-Shot Prompting

Most users start with **Zero-Shot Prompting**: asking a question with no prior examples. While powerful, accuracy increases significantly with **Few-Shot Prompting**, where you provide 2-3 examples of the desired input/output format before the final request. This "in-context learning" is the fastest way to steer a model's style and accuracy without fine-tuning.

> **Key Term - Zero-Shot Prompting:** Providing a prompt to an LLM without any examples of the task. The model relies entirely on its pre-existing training to interpret the instruction.

> **Key Term - Few-Shot Prompting:** Providing a prompt that includes a few examples (input-output pairs) of the task. This "primes" the model to follow the pattern provided in the examples.

### Chain of Thought (CoT)

Chain of Thought forces the model to "think out loud" by generating intermediate reasoning steps before arriving at a final answer. By simply adding the phrase *"Let's think step by step,"* or providing few-shot examples that include reasoning, models become significantly better at arithmetic, symbolic reasoning, and multi-step logic.

> **Key Term - Chain of Thought (CoT):** A prompting technique that encourages the model to generate a sequence of intermediate reasoning steps. This prevents the model from "rushing" to an incorrect answer by providing a workspace for logical deduction.

### Beyond Linear: Tree and Graph of Thought

For truly complex problems, linear reasoning isn't enough.
- **Tree of Thought (ToT):** The model explores multiple reasoning branches simultaneously. If a branch leads to a dead end, the model can backtrack and try another path (using algorithms like BFS or DFS).
- **Graph of Thought (GoT):** The most flexible paradigm. It treats reasoning as a network (graph). Distinct "thoughts" can be merged together, loops can be created to refine ideas, and successful reasoning nodes can be reused across different parts of a problem.

> **Key Term - Tree of Thought (ToT):** A framework where the model generates multiple potential reasoning steps at each point, forming a tree. It evaluates these steps and decides which branches to prune or continue exploring.

> **Key Term - Graph of Thought (GoT):** A framework that models reasoning as a non-linear graph. It allows for advanced operations like merging multiple lines of reasoning into a single conclusion or looping back to iterate on a previous thought.

### Dialog State Management

In a single-turn interaction, you send a prompt and receive a response. But most real-world applications — chatbots, AI assistants, code review tools — are **multi-turn**: the user says something, the model responds, and the conversation continues over many rounds.

**Dialog State** is the mechanism by which you preserve conversational context. Since most LLM APIs are stateless (each API call is independent), you must explicitly pass the conversation history — every prior user message and model response — back into the next prompt.

The standard pattern is to maintain a `messages` list, appending each new exchange before the next call:

```python
conversation_history = [
    {"role": "system", "content": "You are a helpful MLOps assistant."}
]

def chat(user_input: str) -> str:
    # Append the new user message to history
    conversation_history.append({"role": "user", "content": user_input})

    # In practice: send conversation_history to your LLM API here
    # response = call_llm_api(messages=conversation_history)
    assistant_reply = f"[Model response to: {user_input}]"  # placeholder

    # Append the model's reply to maintain full context
    conversation_history.append({"role": "assistant", "content": assistant_reply})
    return assistant_reply
```

> **Key Term - Dialog State:** The accumulated history of a multi-turn conversation, typically stored as a list of `{role, content}` message objects. It must be explicitly managed and passed into each API call, since LLMs have no memory between independent requests. As a conversation grows, dialog state can consume significant context window tokens.

### Prompt Design and Response Curation

Writing a prompt is the beginning, not the end. **Prompt Design** is an iterative engineering process:

1. **Define the output contract first.** Specify exactly what format you want the response in (JSON, numbered list, table) before writing the reasoning instructions. Models are far more consistent when the structure is explicit.
2. **Control length and scope.** Use instructions like *"In 3 bullet points"* or *"In no more than 100 words"* to prevent verbose, off-track responses.
3. **Temperature and token budgets as design tools.** Setting `temperature=0.0` (deterministic) vs. `0.7` (creative) is a design decision that belongs in your prompt specification alongside the text itself.
4. **Curate responses systematically.** When a prompt produces a bad output, diagnose *why*: Was the instruction ambiguous? Did the model lack necessary context? Was the temperature too high? Log prompts and responses in pairs so you can iterate empirically rather than guessing.

> **Key Term - Prompt Engineering:** The discipline of crafting, testing, and iterating on input prompts to reliably guide a model toward a desired output format, style, and accuracy. It is an empirical engineering practice, not a one-shot writing task.

## Code Example

```python
# --- Example: Few-Shot + Chain of Thought ---

prompt = """
Task: Classify if a sentence is 'Enthusiastic' or 'Neutral'. 
Structure your response as: [Reasoning] -> [Classification]

Example 1:
Sentence: "I absolutely cannot wait for the launch of the new SageMaker feature!"
Reasoning: The user uses strong positive adverbs like 'absolutely' and an exclamation mark. 
Classification: Enthusiastic

Example 2:
Sentence: "The model training is scheduled to begin at 4 PM local time."
Reasoning: This is a factual statement regarding a timeline without emotional indicators.
Classification: Neutral

Task:
Sentence: "I guess the new update is okay, but I was expecting more."
Reasoning: 
"""
# The model will now follow the few-shot CoT pattern to generate the reasoning 
# before finally classifying the sentiment.
```

## Additional Resources

- [Language Models are Few-Shot Learners (GPT-3 Paper)](https://arxiv.org/abs/2005.14165)
- [Chain-of-Thought Prompting Elicits Reasoning in LLMs](https://arxiv.org/abs/2201.11903)
- [Graph of Thought: Solving Complex Problems with LLMs](https://arxiv.org/abs/2308.09687)
