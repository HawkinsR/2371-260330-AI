# Lab: Implementing Training Mechanics

## The Scenario
Your team has been given a volatile deep learning model that frequently crashes mid-training due to "exploding gradients" yielding `NaN` losses. Furthermore, the model has a nasty habit of memorizing the training data and overfitting terribly if left to train overnight. Your objective is to refactor the broken training script to include a Gradient Clipping mechanism to bound the explosive math, and an Early Stopping callback to halt training the moment the validation loss begins to rise.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e034-training_mechanics_lab.py`.
3. Read the custom `EarlyStopping` class provided for you. Notice it now accepts **both** `val_loss` and the `model` as arguments so it can save a copy of the best weights when validation improves.
4. Complete the `robust_training_loop` function:
   - Initialize the `EarlyStopping` callback with a `patience` of 3.
   - Inside the training loop batch logic, immediately after `loss.backward()`, use `torch.nn.utils.clip_grad_norm_` to clip the gradients of `model.parameters()` to a `max_norm` of `1.0`.
   - After the optimizer steps, and after you calculate the `val_loss`, pass **both** `val_loss` and `model` to your `early_stopping` object: `early_stopping(val_loss, model)`.
   - Check the `early_stop` boolean attribute on your callback object. If `True`, print a warning and `break` out of the epoch loop to stop training.
5. After the training loop, restore the best model weights:
   - Call `model.load_state_dict(early_stopper.best_weights)` to rewind the model back to the checkpoint that had the lowest validation loss, discarding any weights learned during the overfitting phase.

## Definition of Done
- The script executes successfully.
- Training automatically halts at **Epoch 8** (the simulated val_loss rises for 3 consecutive epochs starting at Epoch 6, triggering the patience counter).
- The text `"Early stopping triggered! Training halted to prevent overfitting"` is printed to the console.
- The final line printed is `"Training complete. Model restored to best checkpoint."`, confirming the weight restoration step ran.
