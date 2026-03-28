# Demo: Linear Regression and Optimizers

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
|---|---|
| **Weights and Biases** | *"In a linear equation `y = mx + b`, what role does `m` play versus `b`? How do those map to weights and biases in a neural network?"* |
| **Forward Pass** | *"If water flows from a source through pipes to a drain, which direction is the 'forward' direction? What happens to the data as it moves through each layer?"* |
| **Gradient Descent** | *"Imagine you're blindfolded on a hilly landscape and want to reach the lowest point. What strategy would you use to navigate downhill?"* |
| **Learning Rate** | *"If there are two hikers going downhill — one taking tiny cautious steps and one taking giant leaps — what could go wrong for each of them?"* |
| **Momentum** | *"Why might a ball rolling downhill move faster over time rather than at a constant speed? How does that benefit an optimization algorithm?"* |

## Phase 1: The Concept (Whiteboard/Diagram)

**Time:** 10 mins

1. Open `diagrams/linear-regression-module.mermaid`.
2. Introduce Object-Oriented AI:
   - "Yesterday, we manually wrote `y = w * x + b`. If we have a billion parameters, doing that manually is impossible. PyTorch utilizes standard Object-Oriented Programming (OOP) to handle this."
   - Point to the `Module` base class. Explain that *every* neural network we ever build inherits from `nn.Module`.
   - Discuss `__init__` (where we define our structural parts) and `forward` (how data travels through those parts).
   - Point to the `TrainingLoop`. The `Optimizer` is a distinct object that holds a reference specifically to the `LinearRegressionModel`'s parameters.

## Phase 2: The Code (Live Implementation)

**Time:** 30 mins

1. Open `code/d040-linear-regression-from-scratch.py`.
2. **The Architecture (Lines 11-19):**
   - Emphasize the `super()` call. Without it, PyTorch breaks.
   - Show how `nn.Linear` replaces our manual weight/bias definition from Day 1. It acts as a black box that holds our parameters for us.
3. **Loss and Optimizers (Lines 30-34):**
   - "MSE Loss compares distance. The optimizer, Adam, is the engine that drives the car based on that distance." Emphasize passing `model.parameters()` to the optimizer so it knows *what* to optimize.
4. **The Training Loop (Lines 40-57):**
   - This is the crux of the class! Walk strictly through the 5 distinct, inviolable steps of the PyTorch training loop:
     1. Forward Pass
     2. Loss Calculation
     3. Zero Gradients
     4. Backward Pass
     5. Optimizer Step
   - *CRITICAL:* Ask the class, "Why do we call `optimizer.zero_grad()`? What happens if we forget it?" (Answer: PyTorch accumulates gradients across batches; the model will explode.)
5. **Execution:**
   - Run the script. Watch the Loss decrease drastically over 200 epochs until the weights converge near `w=20, b=20`. Highlight the final strict prediction without gradient tracking.
