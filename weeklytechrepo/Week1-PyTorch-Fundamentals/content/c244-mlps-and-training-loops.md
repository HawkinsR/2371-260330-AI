# Multi-Layer Perceptrons and Training Loops

## Learning Objectives

- Understand the structure of a Multi-Layer Perceptron (MLP) and the role of activation functions.
- Explain the difference between network width and network depth, and when each matters.
- Identify overfitting and underfitting using the Bias-Variance tradeoff.
- Apply L1, L2, and Dropout regularization techniques to improve generalization.
- Build a complete PyTorch training loop from scratch, step by step.
- Implement a validation loop and save model checkpoints.

## Why This Matters

This file brings together everything from the first week. You are no longer building a single linear layer—you are stacking multiple non-linear layers to create a network that can approximate complex real-world functions. Understanding how to correctly structure the training loop and validate against unseen data is what separates a model that happens to work on your laptop from one that reliably works in production.

## The Concept

### MLPs and Activations

A **Multi-Layer Perceptron (MLP)** consists of an input layer, one or more hidden layers, and an output layer. Data passes through each layer in sequence.

The key insight that makes deep networks powerful: stacking purely linear layers is mathematically equivalent to having just one linear layer—no matter how many you add. To model complex, non-linear functions (like the relationship between pixel values and image labels), you must inject **non-linearity** between layers using an activation function.

**ReLU** (Rectified Linear Unit) is the most commonly used activation function. Its formula is: `f(x) = max(0, x)`. It passes positive values through unchanged and sets negative values to zero. It is computationally cheap and works well in practice.

> **Key Term - Multi-Layer Perceptron (MLP):** A feedforward neural network with at least one hidden layer between the input and output. Each node in a hidden layer applies a weighted sum of its inputs followed by a non-linear activation function.

> **Key Term - Non-linearity:** A mathematical property meaning the relationship between inputs and outputs is not a straight line. Without non-linearity, stacked linear layers collapse into a single linear transformation and cannot model complex data.

> **Key Term - ReLU (Rectified Linear Unit):** An activation function defined as `f(x) = max(0, x)`. It is the default choice for hidden layers because it is simple, fast to compute, and avoids some gradient degradation problems present in older activation functions like Sigmoid.

### Width vs Depth

Two key design decisions shape an MLP's capacity:

- **Width** refers to the number of neurons in a hidden layer. A wider network can represent more features in parallel at a given depth.
- **Depth** refers to the number of hidden layers. A deeper network can learn more hierarchical, abstract representations—but it is also harder to train and more prone to vanishing gradients.

In practice, deeper networks tend to generalize better than very wide shallow ones, which is why modern architectures favor depth over width.

### Overfitting and Regularization

If a model performs well on training data but poorly on new data, it has **overfit**—it memorized the training examples rather than learning general patterns.

The **Bias-Variance Tradeoff** describes this tension:
- **High bias (underfitting):** The model is too simple and misses patterns in both training and test data.
- **High variance (overfitting):** The model is too complex and fits noise in the training data that doesn't generalize.

**Regularization** techniques help keep the model from overfitting:
- **L2 Regularization (Weight Decay):** Adds a penalty proportional to the square of the weights to the loss. This encourages smaller weights overall.
- **L1 Regularization:** Adds a penalty proportional to the absolute value of the weights. This tends to push less important weights to exactly zero, producing a sparse model.
- **Dropout:** During training, randomly sets a fraction of neurons to zero on each forward pass. This forces the network to learn redundant pathways and prevents over-reliance on any individual neuron.

> **Key Term - Bias-Variance Tradeoff:** A fundamental tension in machine learning. A model with too much bias underfits—too simple to capture real patterns. A model with too much variance overfits—too closely tuned to the training data. The goal is to find the right level of complexity that generalizes well.

> **Key Term - L1 Regularization:** A penalty added to the loss equal to the sum of the absolute values of the weights. It encourages sparse weight vectors by pushing unimportant weights toward zero.

> **Key Term - L2 Regularization (Weight Decay):** A penalty added to the loss equal to the sum of the squared weights. It discourages any single weight from becoming very large, producing a more balanced, generalizable model.

### The Training and Validation Loop

The standard PyTorch training loop has five steps, repeated for every batch:

1. **Zero Gradients:** Clear gradients from the previous step with `optimizer.zero_grad()`.
2. **Forward Pass:** Run the batch through the model to get predictions.
3. **Calculate Loss:** Compare predictions to ground truth using the criterion.
4. **Backpropagation:** Call `loss.backward()` to compute gradients via Autograd.
5. **Optimizer Step:** Call `optimizer.step()` to update the weights.

After each training epoch, run a **validation loop** with `torch.no_grad()` and `model.eval()`. Disabling gradient tracking saves memory and ensures Dropout is turned off during evaluation. If the validation loss improves, save a checkpoint.

> **Key Term - Epoch:** One complete pass through the entire training dataset. Each epoch consists of many mini-batch steps.

> **Key Term - Checkpoint:** A saved snapshot of the model's weights at a specific point during training. Checkpointing lets you resume interrupted training runs and restore the best-performing version of the model.

## Code Example

```python
import torch
import torch.nn as nn

class BasicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Hidden layer: 768 input features -> 128 neurons
        self.fc1 = nn.Linear(768, 128)
        self.relu = nn.ReLU()
        # Dropout: randomly zero 50% of neurons during each training forward pass
        self.dropout = nn.Dropout(p=0.5)
        # Output layer: 128 neurons -> 10 class scores
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply linear layer then ReLU
        x = self.dropout(x)          # Apply dropout
        return self.fc2(x)           # Final output (no activation — handled by loss fn)

# --- Standard 5-step training loop ---
# for batch_x, batch_y in dataloader:
#     optimizer.zero_grad()                   # 1. Clear old gradients
#     predictions = model(batch_x)            # 2. Forward pass
#     loss = criterion(predictions, batch_y)  # 3. Calculate loss
#     loss.backward()                         # 4. Backpropagate
#     optimizer.step()                        # 5. Update weights
```

## Additional Resources

- [Training a Classifier Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Dropout Explained](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
