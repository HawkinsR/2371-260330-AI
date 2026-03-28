# Linear Models and Optimizers

## Learning Objectives

- Design custom neural network architectures using `nn.Module`.
- Implement a model's forward pass using the `forward()` method.
- Understand when to use MSE loss vs. CrossEntropy loss.
- Train a model using SGD and Adam optimizers.
- Use a Learning Rate Scheduler to decay the learning rate during training.

## Why This Matters

Building on our understanding of tensors, we now construct our first predictive models. The `nn.Module` class is the foundation of every PyTorch model, from a two-line linear regression to a billion-parameter language model. Choosing the right loss function and optimizer directly determines whether your model converges to a good solution or wastes compute spinning in place.

## The Concept

### The `nn.Module` and the Forward Pass

Every PyTorch model inherits from `torch.nn.Module`. In the `__init__` method, you define the layers and parameters (weights and biases) the model will use. In the `forward()` method, you define exactly how data flows through those layers to produce an output prediction.

PyTorch's `nn.Module` automatically tracks all `nn.Parameter` objects registered inside it, which makes it easy to pass them to an optimizer with a single call to `model.parameters()`.

> **Key Term - Weights and Biases:** The learnable parameters of a neural network. A **weight** is a multiplier that controls how strongly an input influences an output. A **bias** is an additive offset that allows the model to shift its output even when all inputs are zero. Together, they are what the optimizer adjusts during training.

> **Key Term - Forward Pass:** The process of feeding input data through the network's layers to produce a prediction. Data flows in one direction: from the input layer, through any hidden layers, to the output. The forward pass always happens before backpropagation.

### Loss Functions

A loss function measures how wrong the model's predictions are. The choice of loss function depends on the task:

- **Regression (predicting continuous values):** Use `nn.MSELoss()` (Mean Squared Error), which penalizes large errors more heavily than small ones.
- **Classification (predicting categories):** Use `nn.CrossEntropyLoss()`, which penalizes confident wrong predictions most severely.

> **Key Term - Loss Function (Criterion):** A mathematical function that compares the model's predicted output to the true target value and returns a single number representing the error. The goal of training is to minimize this number.

### Optimizers and Schedulers

Once gradients are computed by Autograd, the **optimizer** uses them to update the model's weights.

- **SGD (Stochastic Gradient Descent):** The simplest optimizer. It adjusts weights by taking a step proportional to the gradient. Straightforward but can be slow to converge on complex problems.
- **Adam:** A more advanced optimizer that tracks a running average of past gradients (momentum) and adapts the learning rate individually for each parameter. Adam tends to converge faster and is the default choice for most deep learning tasks.

**Schedulers** automatically reduce the learning rate over the course of training. A large learning rate helps early on (fast exploration), while a smaller one helps later (fine-tuned convergence).

> **Key Term - Optimizer:** The algorithm responsible for updating a model's weights after each gradient calculation. It determines the direction and magnitude of each weight update.

> **Key Term - Epoch:** One complete pass through the entire training dataset. Models typically require many epochs—sometimes dozens or hundreds—to fully converge.

> **Key Term - Learning Rate:** A hyperparameter that controls the size of the steps the optimizer takes when updating weights. If it is too large, the optimizer overshoots the minimum and the model fails to **converge**. If it is too small, training is extremely slow or gets stuck. Common starting values are `0.01` or `0.001`.

> **Key Term - Gradient Descent:** The core optimization strategy: repeatedly compute the gradient of the loss with respect to model weights, then take a step in the direction that reduces the loss. "Stochastic" means we compute the gradient on a small random mini-batch rather than the full dataset, which is much faster per step.

> **Key Term - Momentum:** A technique used by optimizers like Adam to accelerate training. Instead of updating weights based solely on the current gradient, momentum blends the current gradient with a running average of past gradients. This smooths out noisy updates and helps the optimizer push through flat regions of the loss landscape.

## Code Example

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple linear regression model
class SimpleLinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # nn.Linear creates a layer with learnable weights and biases
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Data flows directly through the linear layer
        return self.linear(x)

# Instantiate a model that takes 1 input feature and predicts 1 value
model = SimpleLinearRegression(input_size=1, output_size=1)

# Use MSE loss for a regression task
criterion = nn.MSELoss()

# Adam optimizer — pass all model parameters so it knows what to update
optimizer = optim.Adam(model.parameters(), lr=0.1)

# StepLR scheduler: multiply the learning rate by 0.1 every 10 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

## Additional Resources

- [Building Neural Networks in PyTorch](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [PyTorch Optim Library](https://pytorch.org/docs/stable/optim.html)
