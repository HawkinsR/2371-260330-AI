# PyTorch Tensors and Autograd

## Learning Objectives

- Differentiate between Supervised and Unsupervised learning within the ML Lifecycle.
- Understand the relationship between PyTorch Tensors and NumPy arrays and how to convert between them.
- Manipulate tensor shapes, data types, and perform mathematical operations.
- Move tensors between CPU and GPU hardware using `.to(device)` and CUDA.
- Understand how Autograd builds and traverses a Computational Graph to calculate gradients.

## Why This Matters

Tensors are the primary data structure of all deep learning. Every image, sentence, and numeric dataset you feed into a neural network must first be represented as a tensor. Understanding how PyTorch tracks operations on tensors—and automatically calculates the gradients needed to train a model—is the bedrock skill the rest of this curriculum builds on.

> **Key Term - Hardware Acceleration:** Using specialized processors (GPUs or TPUs) to run mathematical operations much faster than a standard CPU. Deep learning requires billions of matrix multiplications, and GPUs are built with thousands of small, parallel cores specifically designed for this kind of workload.

> **Key Term - Gradient:** The derivative of a loss function with respect to a model's parameters. It tells us how much each parameter contributed to the current error, and in which direction to adjust it to reduce that error. The optimizer uses gradients to update weights during training.

## The Concept

### The ML Lifecycle & Learning Paradigms

The machine learning lifecycle is an iterative process: collect data, preprocess it, design a model architecture, train it, evaluate its performance, and deploy it. Then repeat as you gather more data or discover new failure modes.

Within this lifecycle, two fundamental learning paradigms define how a model learns:
- **Supervised Learning:** Every training example has a known label. You teach the model by showing it labeled examples (e.g., images already tagged as "cat" or "dog") and training it to predict those labels on new data.
- **Unsupervised Learning:** The training data has no labels. The model must find structure on its own—grouping similar data points into clusters or learning compressed representations.

### Tensors vs NumPy

A **tensor** is a multi-dimensional array of numbers. A scalar is a 0-dimensional tensor, a vector is 1-dimensional, and a matrix is 2-dimensional. PyTorch tensors behave almost identically to NumPy arrays—same indexing, same math operations—but with two critical additions:

1. **GPU Support:** A tensor can be moved to a GPU with a single line of code, enabling massive speedups for matrix math.
2. **Automatic Differentiation:** PyTorch tracks every operation performed on a tensor so it can compute gradients automatically during backpropagation.

> **Key Term - Tensor:** A multi-dimensional array of numerical data that serves as the fundamental data structure for neural networks. Unlike a NumPy array, a PyTorch tensor can live on a GPU and participate in automatic gradient computation.

### Operations, Shapes, and Devices

Every tensor has a **shape** (e.g., `[32, 3, 224, 224]` for a batch of 32 RGB images) and a **dtype** (e.g., `float32` for model weights, `int64` for class labels). When you move a tensor to the GPU with `.to('cuda')`, all subsequent mathematical operations on it run on the GPU. PyTorch will raise an error if you try to perform operations on two tensors that are on different devices.

### Autograd Mechanics

When you perform operations on a tensor with `requires_grad=True`, PyTorch silently records each operation in a running history called the **autograd tape**. This history forms a Directed Acyclic Graph (DAG). When you call `.backward()` on your final loss value, PyTorch walks this graph in reverse, applying the chain rule of calculus at each node to compute the gradient of the loss with respect to every parameter. Those gradients are stored in the `.grad` attribute of each leaf tensor.

> **Key Term - Directed Acyclic Graph (DAG):** A data structure made of nodes connected by directed arrows with no cycles (no loops). In PyTorch's autograd system, each node represents a tensor operation (like addition or matrix multiply), and the arrows trace how data flowed from inputs to outputs. "Acyclic" ensures gradients can always be traced backward without getting stuck in an infinite loop.

> **Key Term - Leaf Node:** A tensor that was created directly by the developer (not as the result of another operation). When `.backward()` is called, PyTorch stores gradients only on leaf nodes. Intermediate nodes are discarded by default to save memory.

> **Key Term - Autograd:** PyTorch's automatic differentiation engine. It records a "tape" of all operations performed on tracked tensors during the forward pass, then replays that tape in reverse during `.backward()` to compute gradients automatically—eliminating the need to derive and implement calculus by hand.

## Code Example

```python
import torch
import numpy as np

# --- NumPy to PyTorch Bridge ---
np_array = np.array([[1, 2], [3, 4]])
tensor = torch.from_numpy(np_array).float()

# --- Reshaping ---
# .view(-1) flattens the tensor into a 1D array
flat_tensor = tensor.view(-1)  # shape: [4]

# --- Moving to GPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = tensor.to(device)

# --- Autograd in Action ---
# requires_grad=True tells PyTorch to track operations on this tensor
x = torch.tensor(2.0, requires_grad=True)

# Perform a forward operation
y = x ** 2  # y = x^2

# Trigger backpropagation — autograd traverses the DAG and computes dy/dx
y.backward()

# The derivative of x^2 is 2x. At x=2.0, the gradient is 4.0
print(x.grad)  # Output: tensor(4.)
```

## Additional Resources

- [PyTorch Tensors Tutorial](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [Autograd Engine Documentation](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
