# Demo: Model Artifacts and Inference

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
|---|---|
| **Model Artifact** | *"A trained model's weights file `.pth` is just a file full of numbers. Why can't you load it without also having the Python class (the architecture) that defines what those numbers mean?"* |
| **Serialization / JSON** | *"What does it mean to 'serialize' data? If your web app sends a Python dictionary over an HTTP request, what format does it need to be converted to first, and what converts it back on the server?"* |
| **Inference Hook** | *"What's the difference between a function you call directly and a 'hook' that gets called automatically by a framework? Why would SageMaker use hooks (`model_fn`, `predict_fn`) instead of just having you write a `main()` function?"* |
| **Eval Mode (`model.eval()`)** | *"Dropout randomly zeroes out neurons during training to prevent overfitting. Why would you absolutely never want Dropout active when you're serving real-time predictions to users? What does `.eval()` do?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/model-inference-packaging.mermaid`.
2. Trace the path from the end of the Training Job that drops a `.pth` file. Show how this file is mathematically useless without the Python class architecture defining it. This is why we zip them together into `model.tar.gz`.
3. Walk through the **SageMaker Endpoint Initialization** subgraph. Explain that `model_fn` only runs once to spare compute resources. It effectively caches the heavy weights in RAM.
4. **Discussion:** Transition to the **Real-Time Inference** subgraph. Ask the class: "If your React application sends an HTTP POST request, does the `predict_fn` know how to read JSON strings natively?" (Answer: No, PyTorch uses Tensors. We must use `input_fn` to translate strings into math, and `output_fn` to translate math back into strings).

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d052-serving-and-inference.py`.
2. Walk through the four required hooks. 
   - Notice the `.eval()` and the `.requires_grad = False` loop inside the `model_fn()`. Emphasize that failing to set Eval mode means layers like Dropout will remain active, causing wildly inaccurate and randomized predictions during inference!
   - Highlight the precise mapping inside `input_fn()` where `json.loads` converts the string into a nested Python list, which is then mapped to `torch.tensor`.
3. Execute the script via `simulate_endpoint_lifecycle()`. 
4. The output logs uniquely tag each step of the pipeline. Show the student the transition from `[Endpoint Server Start]` to `[HTTP Request Received]`.

## Summary
Reiterate that mastering Cloud Inference means understanding serialization and deserialization; a Data Scientist's model is only as powerful as their API interface converting human requests into PyTorch matrices.
