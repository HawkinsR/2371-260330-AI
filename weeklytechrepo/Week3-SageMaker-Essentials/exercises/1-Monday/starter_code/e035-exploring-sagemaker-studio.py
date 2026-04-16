# e035-exploring-sagemaker-studio.py
# Lab: Exploring SageMaker Studio — Prompt Engineering Comparison
#
# NOTE: Studio Domain, User Profile, and IAM Role setup are configured in the
# AWS Console. This script covers Task 3: the prompt engineering comparison.
# In a real environment, the simulate_model_call() function would be replaced
# with a SageMaker JumpStart predictor.predict() call.

def build_zero_shot_prompt(task_description: str) -> str:
    """
    Task 3a: Build a Zero-Shot Prompt.
    A zero-shot prompt gives the model ONLY the task instruction — no examples.
    """
    # TODO: Return a single prompt string that instructs the model to perform
    # the task described by task_description with NO examples.
    # Recommended structure:
    #   "Task: {task_description}\n\nAnswer:"
    return None


def build_few_shot_prompt(task_description: str, examples: list) -> str:
    """
    Task 3b: Build a Few-Shot Prompt.
    A few-shot prompt includes 2-3 input/output examples before the final task.
    
    Args:
        task_description (str): The task the model should perform.
        examples (list): A list of dicts with keys 'input' and 'output'.
    """
    # TODO: Build a complete prompt string by:
    # 1. Iterating through each dict in `examples`
    # 2. Formatting each as:  "Input: {example['input']}\nOutput: {example['output']}\n\n"
    # 3. Appending the final task instruction at the end:
    #    "Task: {task_description}\n\nInput: [new review here]\nOutput:"
    return None


def simulate_model_call(prompt: str, model_name: str = "meta-textgeneration-llama-3-8b") -> str:
    """
    Simulates a JumpStart endpoint invocation for local testing.
    In the real lab environment, replace this body with:
        predictor = JumpStartModel(model_id=model_name).deploy(...)
        response = predictor.predict({"inputs": prompt})
    """
    if prompt is None:
        return "[ERROR: Prompt was None — complete the TODO before calling this function]"

    word_count = len(prompt.split())
    # Heuristic: more context in the prompt → higher simulated quality score
    quality = "HIGH" if word_count > 25 else "MEDIUM" if word_count > 10 else "LOW"
    return (
        f"[Simulated JumpStart Response]\n"
        f"  Model   : {model_name}\n"
        f"  Input words: {word_count}  |  Estimated output quality: {quality}\n"
        f"  Result  : 'POSITIVE'  (confidence: 0.91)"
    )


def compare_prompting_strategies():
    """
    Task 3c: Compare Zero-Shot vs Few-Shot results and record observations.
    """
    print("=" * 60)
    print("  JumpStart Prompt Engineering Comparison")
    print("=" * 60)

    task = "Classify the sentiment of a customer product review as POSITIVE or NEGATIVE."

    examples = [
        {"input": "This laptop exceeded all my expectations. Absolutely love it!",
         "output": "POSITIVE"},
        {"input": "Stopped working after two days. Terrible build quality.",
         "output": "NEGATIVE"},
        {"input": "Shipping was slow but the product itself is decent.",
         "output": "POSITIVE"},
    ]

    # --- Zero-Shot ---
    zero_shot = build_zero_shot_prompt(task)
    print("\n=== Zero-Shot Prompt ===")
    if zero_shot is not None:
        print(zero_shot)
        print("\n--- Simulated Model Response ---")
        print(simulate_model_call(zero_shot))
    else:
        print("[Incomplete] Finish the TODO in build_zero_shot_prompt().")

    # --- Few-Shot ---
    few_shot = build_few_shot_prompt(task, examples)
    print("\n=== Few-Shot Prompt ===")
    if few_shot is not None:
        print(few_shot)
        print("\n--- Simulated Model Response ---")
        print(simulate_model_call(few_shot))
    else:
        print("[Incomplete] Finish the TODO in build_few_shot_prompt().")

    # --- Your Observations ---
    # TODO: After running the script, answer the following in comments below:
    #
    # Q1. Looking at the word counts, which prompt gives the model more context?
    # A1: 
    #
    # Q2. In what real-world scenario would zero-shot be good enough?
    # A2: 
    #
    # Q3. What makes a good few-shot example pair? Why did we choose 3 examples?
    # A3: 
    print("\n[Reminder] Complete the observation comments in compare_prompting_strategies().")


if __name__ == "__main__":
    compare_prompting_strategies()
