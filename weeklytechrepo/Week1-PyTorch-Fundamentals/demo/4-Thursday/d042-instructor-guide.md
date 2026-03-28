# Demo: MLPs and Complex Training Loops

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
|---|---|
| **Non-linearity / ReLU** | *"Can you draw a straight line through a figure like a spiral or a circle to perfectly separate two colors? What kind of boundary would you need instead?"* |
| **Bias-Variance Tradeoff** | *"What's the difference between a student who memorized last year's exam answers vs. one who truly understood the subject? Which one would struggle on a new exam?"* |
| **L1 / L2 Regularization** | *"If a basketball coach penalizes players for over-relying on just one move, how does that make the team more robust? How does this mirror what L1/L2 does to model weights?"* |
| **Epoch** | *"If you were training for a marathon and had to run a 10-mile course repeatedly, what would one 'epoch' be? Why run it more than once?"* |
| **Checkpoint** | *"If you were playing a long video game with no save points and your power cut out, what would you lose? How does model checkpointing solve that same problem?"* |

## Phase 1: The Concept (Whiteboard/Diagram)

**Time:** 15 mins

1. Open `diagrams/training-validation-loop.mermaid`.
2. Address the two distinct tracks:
   - "A single epoch isn't just training. It is Training *followed immediately by* Validation."
   - Walk through the red `Training Phase` bucket. Reinforce the 5 inviolable steps of PyTorch training (Forward, Loss, Zero, Backward, Step) that we learned on Tuesday.
   - Walk through the blue `Validation Phase` bucket. Highlight the two mandatory flags: `model.eval()` and `torch.no_grad()`.
   - Ask the class: "Why don't we call `loss.backward()` in the validation phase?" (Answer: We use validation data to *test* the current iteration of the model. If we backpropagate, we are cheating by letting the model memorize the answers to the test).
   - "If the test looks good, we hit the green `torch.save` node to lock in our progress into a file."

## Phase 2: The Code (Live Implementation)

**Time:** 25 mins

1. Open `code/d042-complete-mlp-training-loop.py`.
2. **The MLP Architecture (Lines 12-25):**
   - We are adding layers. We now have `fc1` and `fc2`.
   - Point to `nn.ReLU()`. "Without a non-linear activation function, two linear layers map back to a single linear layer. We need this to learn curves."
   - Point to `nn.Dropout(p=0.5)`. "We randomly kill 50% of the neurons every pass so the network doesn't memorize the training data. It prevents overfitting."
3. **The Architecture of the Loop (Lines 35-51):**
   - Point explicitly to line 38: `model.train()`. "This turns Dropout ON."
   - Point explicitly to line 48: `model.eval()`. "This turns Dropout OFF."
   - Point explicitly to line 51: `with torch.no_grad()`. "This turns off the massive RAM overhead of the Autograd engine, speeding up our validation pass considerably."
4. **Checkpointing (Lines 57-67):**
   - Discuss tracking `best_val_loss`.
   - Emphasize `torch.save(model.state_dict())`. We do *not* save the `model` object itself because if we rename the class file later, the model breaks. Save the lightweight dictionary of raw numbers.
5. **Execution:**
   - Run the script and observe the console output.
   - Wait for strings saying "Validation loss decreased! Saving checkpoint...". Show the class the physical `checkpoints/best_model.pth` file written to disk.
