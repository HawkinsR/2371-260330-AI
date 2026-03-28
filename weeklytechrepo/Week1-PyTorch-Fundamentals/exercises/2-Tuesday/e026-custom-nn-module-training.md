# Lab: Custom NN Module and Training

## The Scenario
You've been tasked with building a predictive model for a small e-commerce site. They have a dataset mapping a user's 'Time on Site' (in minutes) to whether they made a purchase (1 for yes, 0 for no). You need to construct a custom PyTorch Neural Network module to classify these users, train it using a classification loss function, and implement a learning rate scheduler to ensure smooth convergence.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e026-classification_lab.py`.
3. Complete the `PurchaseClassifier` class:
   - Inherit correctly from `nn.Module`.
   - Initialize a single linear layer in the constructor (`__init__`) that maps 1 input feature to 1 output feature.
   - Implement the `forward` pass to route the input through the linear layer.
4. Complete the `train_classifier` function:
   - Instantiate your `PurchaseClassifier`.
   - Define `BCEWithLogitsLoss` as the criterion (Binary Cross Entropy since we are classifying 0 or 1).
   - Use the `Adam` optimizer with an initial learning rate of `0.5`.
   - Define a `StepLR` scheduler that multiplies the learning rate by `0.5` every `20` epochs.
   - Core Training Loop: Write the 5 essential steps of PyTorch training (forward, loss, zero_grad, backward, step) inside the epoch loop.
   - Step the scheduler at the end of each epoch.

## Definition of Done
- Your script runs and loss decreases over time.
- The model correctly outputs a higher numerical logit for '10.0' minutes on site compared to '0.5' minutes on site during the final testing phase.
- The learning rate decays, which you can verify by printing `scheduler.get_last_lr()` every 20 epochs.
