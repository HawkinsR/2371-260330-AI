# CNN Architecture and Feature Maps

## Learning Objectives

- Build an intuition for how Convolutional Neural Networks (CNNs) process visual data differently than standard MLPs.
- Understand how convolutions work using `nn.Conv2d`, filters, and kernels.
- Explain how Pooling and Stride reduce the size of feature maps between layers.
- Visualize feature maps to understand what patterns a network has learned to detect.
- Describe the problem ResNet was designed to solve and how residual connections address it.

## Why This Matters

Standard MLPs expect a flat list of numbers. When you flatten a 2D image into a 1D array, you lose all information about which pixels are adjacent to each other—an "edge" (two dark pixels side by side) becomes meaningless. CNNs solve this by processing small, localized patches of an image at a time, preserving the 2D structure. This is why CNNs are the standard architecture for any task involving images, from medical scans to self-driving vehicles.

## The Concept

### Convolutions and Filters

A **convolution** works by sliding a small matrix of numbers—called a **filter** or **kernel**—across the entire input image. At each position, it multiplies the filter values against the underlying pixel values and sums them up, producing a single number. The filter then moves one step and repeats.

Each filter detects one specific visual pattern. Early filters learn to detect simple things like horizontal edges or color gradients. Deeper filters combine those simple detections to recognize more complex shapes, like wheels or faces.

> **Key Term - Spatial Context:** The meaningful relationship between pixels based on their position relative to each other. Two dark pixels side-by-side encode an "edge." Flattening an image into a 1D array loses this structure—flattened pixels have no neighbors. CNNs preserve spatial context by always processing local 2D regions together.

> **Key Term - Convolutional Filter (Kernel):** A small matrix of learned weights (typically 3×3 or 5×5) that slides across an image. At each position, it multiplies its values against the patch of pixels underneath it and sums the results into one number. One filter learns one pattern. A layer with 64 filters learns 64 different patterns simultaneously.

### Pooling and Stride

As an image passes through multiple convolutional layers, the computational cost grows quickly. Two techniques control this:

- **Stride:** Instead of the filter moving one pixel at a time, it jumps by `stride` pixels. A stride of 2 cuts the output dimensions in half.
- **Max Pooling:** Looks at a small region (e.g., 2×2 pixels) and keeps only the maximum value, discarding the rest. This reduces the size of the output while retaining the strongest detected features.

Together, these reduce the spatial size of the data as it moves deeper into the network.

> **Key Term - Feature Map:** The output of a convolutional layer. It is a grid where each cell represents how strongly the filter's pattern was present at that location in the input. A layer with 16 filters produces 16 feature maps—one per filter.

> **Key Term - Stride:** The number of pixels the filter moves with each step. A stride of 1 gives dense coverage; a stride of 2 skips every other position and halves the spatial dimensions of the output.

> **Key Term - Translational Invariance:** A model's ability to recognize an object regardless of where it appears in the image. Max Pooling helps achieve this by discarding the exact location of a feature and only recording whether it was detected nearby.

### Visualizing Feature Maps

A key technique for understanding what a CNN has learned is to extract and plot its intermediate feature maps. After passing an image through a convolutional layer, each of the resulting feature maps can be rendered as a grayscale image. Bright areas indicate where that filter's pattern was strongly detected.

This is done in practice using a **forward hook**—a callback that captures the output of a specific layer during inference without modifying the model.

### ResNet and Residual Connections

As networks get deeper, a problem emerges: gradients shrink exponentially as they are multiplied through dozens of layers during backpropagation. By the time they reach the early layers, they are effectively zero—those layers stop learning. This is the **Vanishing Gradient Problem**.

ResNet (Residual Network) solves this by adding **skip connections**: each block passes its raw input directly to the output, bypassing the intermediate layers. Even if the intermediate layers do nothing, the gradient can still flow through the skip connection unchanged. This makes it practical to train networks with hundreds of layers.

> **Key Term - Vanishing Gradient Problem:** During backpropagation, gradients are multiplied together at every layer. In deep networks, these multiplications can shrink the gradient to near-zero before it reaches the early layers, which then stop updating. This prevents deep networks from learning effectively.

## Code Example

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class SimpleCNNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 input channels (RGB image), 16 output feature maps, 3x3 filter
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # Reduces spatial dimensions by half (224 -> 112)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Input:  [Batch, 3, 224, 224]
        x = self.conv1(x)
        x = self.relu(x)
        # After conv: [Batch, 16, 224, 224]
        x = self.pool(x)
        # After pool: [Batch, 16, 112, 112]
        return x

# --- Test the model and visualize feature maps ---
model = SimpleCNNBlock()
dummy_image = torch.randn(1, 3, 224, 224)
feature_maps = model(dummy_image)

print("Output Feature Map Shape:", feature_maps.shape)
# Output: torch.Size([1, 16, 112, 112])

# Visualize the first 4 of the 16 feature maps
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    # detach() removes the tensor from the gradient graph for plotting
    axes[i].imshow(feature_maps[0, i].detach(), cmap='viridis')
    axes[i].set_title(f'Filter {i+1}')
    axes[i].axis('off')
plt.suptitle('Feature Maps from Conv Layer 1')
plt.show()
```

## Additional Resources

- [PyTorch Conv2d Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [Understanding Convolutions Visualized](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)
- [Deep Residual Learning for Image Recognition (ResNet Paper)](https://arxiv.org/abs/1512.03385)
