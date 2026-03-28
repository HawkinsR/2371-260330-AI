# Demo: Advanced Training Mechanics

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
|---|---|
| **Random Seed** | *"If two scientists run the same experiment but get different results each time they try, is it a reliable experiment? How does 'setting a seed' make machine learning experiments more scientific?"* |
| **Early Stopping** | *"If students study harder right before a final exam than the material requires, they might 'overfit' to last year's questions. What signal would a teacher use to tell them to stop studying?"* |
| **Mixed Precision (FP16/FP32)** | *"Why might an architect draw rough sketches in pencil (less precise) for brainstorming but switch to detailed ink drawings for final blueprints? How does this map to training vs. updating weights?"* |
| **Gradient Clipping** | *"If a car's gas pedal could accidentally slam to 100% any time there's a bump, what safety mechanism would you want installed? What's the AI equivalent?"* |
| **Black Box / SHAP** | *"If an AI rejects your loan application, what legal and ethical obligation should the bank have to explain why? How might SHAP help fulfill that obligation?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/training-loop-mechanics.mermaid`.
2. Trace the optimization path. Explain why AMP requires the `autocast` forward pass and the `scaler.scale` backing pass (because 16-bit math can sometimes round small gradients down to zero/underflow; the scaler artificially multiplies them out of danger).
3. Point out the `clip_grad_norm_` block. 
4. **Discussion:** Ask the class: "If Gradient Clipping protects against exploding gradients hitting infinity, what does Early Stopping protect against?" (Answer: Overfitting to the training dataset).

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d048-mixed-precision-and-callbacks.py`.
2. Walk through the `set_seed()` function. Explain that standardizing randomness is essential for debugging and validating model iterations.
3. Review the `EarlyStopping` class. Show how it tracks the `best_loss` and increments the `counter` if the current validation loss is worse.
4. Dive into the training loop inside `demonstrate_advanced_mechanics()`.
   - Point out the `with autocast():` context manager and the subsequent `scaler.unscale_()` call.
   - Show where `clip_grad_norm_` is applied explicitly before the optimizer steps.
5. Execute the script. 
6. Watch the terminal output. Point out when the validation loss begins to "degrade" and the `EarlyStopping counter` kicks in until the script halts.

## Summary
Reiterate that mastering these robust, defensive mechanics marks the transition from amateur scripting to professional, production-level Deep Learning engineering.
