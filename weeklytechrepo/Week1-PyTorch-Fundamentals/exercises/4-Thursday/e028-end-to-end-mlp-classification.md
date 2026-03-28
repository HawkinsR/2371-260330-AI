# Lab: End-to-End MLP Classification

## The Scenario
Your company works with IoT sensor data, and you've been asked to build the core classification engine for a new hardware device. The dataset is already pre-processed and fed to you as Tensors. Your job is to build a Multi-Layer Perceptron (MLP) robust enough to generalize, and to write the full training and validation loops. The hardware engineers need the absolute best model weights saved at the end so they can deploy them to the edge device.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e028-mlp_classification_lab.py`.
3. Complete the `SensorMLP` class:
   - Inherit from `nn.Module`.
   - Build a 3-layer architecture:
     - Input layer (20 features -> 64 hidden units).
     - Hidden layer (64 units -> 32 hidden units).
     - Output layer (32 units -> 4 classes).
   - Apply `nn.ReLU()` activations after the input and hidden layers.
   - Apply `nn.Dropout(p=0.3)` after the `ReLU` activations to prevent overfitting.
4. Complete the `train_and_validate` function:
   - Define `CrossEntropyLoss` and the `Adam` optimizer (learning rate = 0.01).
   - Write the Training Phase loop (remember to set `model.train()`).
   - Write the Validation Phase loop (remember to set `model.eval()`, use `torch.no_grad()`, and do *not* run the optimizer).
   - Implement the Checkpointing logic: Every time the validation loss reaches a new low, print a message and save the `model.state_dict()` to `best_sensor_model.pth`.

## Definition of Done
- The network successfully processes the data without shape errors (check your input/output dimensions).
- The terminal output shows both the Training Loss and Validation Loss for each epoch.
- A file named `best_sensor_model.pth` is created.
- The training accurately utilizes `dropout` during the training phase, but turns it off during the validation phase.
