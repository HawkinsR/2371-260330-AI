# CNN and Module Architecture

## Learning Objectives

- Build an intuition for how Convolutional Neural Networks (CNNs) process visual data differently than standard MLPs.
- Understand how convolutions work using `nn.Conv2d`, filters, and kernels.
- Explain how Pooling and Stride reduce the size of feature maps between layers.
- Visualize feature maps to understand what patterns a network has learned to detect.
- Describe the problem ResNet was designed to solve and how residual connections address it.
- Understand the `nn.Module` class structure and the `forward()` method logic.

## Why This Matters

Standard MLPs expect a flat list of numbers. When you flatten a 2D image into a 1D array, you lose all information about which pixels are adjacent to each other—an "edge" (two dark pixels side by side) becomes meaningless. CNNs solve this by processing small, localized patches of an image at a time, preserving the 2D structure. This is why CNNs are the standard architecture for any task involving images, from medical scans to self-driving vehicles.

## The Concept

### MLPs vs. CNNs: Why Flat Numbers Fail on Images

A **Multi-Layer Perceptron (MLP)** — the fully-connected feedforward network you learned in Week 1 — accepts a flat 1D vector of numbers. To feed an image into an MLP, you must first *flatten* it: a 224×224 RGB image becomes a single vector of 150,528 numbers.

This causes two critical problems:

1. **Spatial information is destroyed.** After flattening, the model has no idea which pixel was adjacent to which. A horizontal edge (two dark pixels side by side) is indistinguishable from the same pixels in random positions. The model cannot learn that *proximity matters*.
2. **Parameter explosion.** The first layer of an MLP connecting 150,528 inputs to just 1,024 neurons requires over **154 million parameters** — before the network even does any useful work. This is impractical to train.

CNNs solve both problems by processing the image in small spatial patches, always respecting the 2D grid structure, and sharing the same filter *weights* across all positions in the image. A single 3×3 filter with 27 parameters can scan an entire 224×224 image — a 5,500× reduction in parameters over the equivalent MLP connection.

> **Key Term - Parameter Sharing:** A CNN uses the same filter weights at every position in the image. This means a filter that detects a horizontal edge in the top-left corner of an image will also detect the same edge in the bottom-right corner using the *same 9 weights* — no extra parameters required. This is the core efficiency mechanism of CNNs.

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

### The `nn.Module` Structure

In PyTorch, all neural network components inherit from `nn.Module`. This class provides the backbone for managing weights, moving models to GPUs, and handling the training process.

1.  **`__init__(self)`**: This is where you define the layers of your network (convolutions, pooling, activation functions). By assigning them as attributes (e.g., `self.conv1 = ...`), PyTorch automatically tracks their parameters.
2.  **`forward(self, x)`**: This is where you define the "flow" of data. You decide how the layers you defined in `__init__` are connected. You do not call `forward` directly; instead, you call the model object like a function (e.g., `output = model(input)`), and PyTorch handles the execution and gradient tracking.

> **Key Term - Modular Design:** Breaking a complex network into smaller, reusable blocks (like a ResNet block). Each block is its own `nn.Module` that can be stacked to build massive architectures.

## Code Example

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# Set this to the path of any JPEG or PNG image on your machine.
# Leave it as None to fall back to a randomly generated dummy image.
#
# Example:  IMAGE_PATH = "C:/Users/you/Pictures/cat.jpg"
# =============================================================================
IMAGE_PATH = None


# --- CNN Block Definition ---
class SimpleCNNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 input channels (RGB), 16 output feature maps, 3x3 filter
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # MaxPool halves the spatial dimensions (224 → 112)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # x shape in:  [Batch, 3,  224, 224]
        x = self.conv1(x)   # → [Batch, 16, 224, 224]
        x = self.relu(x)
        x = self.pool(x)    # → [Batch, 16, 112, 112]
        return x


# --- Load image or fall back to a dummy tensor ---
if IMAGE_PATH:
    from PIL import Image
    from torchvision import transforms

    # torchvision.transforms preprocesses a PIL image into a normalised float tensor.
    # Resize ensures the image matches the 224x224 input size CNNs expect.
    # ToTensor converts PIL [H, W, C] uint8 (0–255) → torch [C, H, W] float32 (0.0–1.0)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    raw_image = Image.open(IMAGE_PATH).convert("RGB")  # Force 3-channel RGB
    input_tensor = preprocess(raw_image).unsqueeze(0)  # Add batch dim: [1, 3, 224, 224]
    source_label = f"Your image: {Path(IMAGE_PATH).name}"
    print(f"Loaded:  {IMAGE_PATH}")
else:
    input_tensor = torch.randn(1, 3, 224, 224)
    source_label = "Source: random noise (no IMAGE_PATH set)"
    print("IMAGE_PATH is None — using a random dummy tensor.")
    print("Set IMAGE_PATH at the top of this file to use your own image.\n")

# --- Run the CNN block ---
model = SimpleCNNBlock()
model.eval()  # Disable any training-specific behaviour (good inference habit)

with torch.no_grad():  # No gradient tracking needed for visualisation
    feature_maps = model(input_tensor)

print(f"Input tensor shape  : {input_tensor.shape}  [batch, channels, H, W]")
print(f"Feature maps shape  : {feature_maps.shape}  [batch, filters, H, W]")
print(f"  → {feature_maps.shape[1]} filters, each {feature_maps.shape[2]}×{feature_maps.shape[3]} pixels")

# --- Visualise: original image + first 4 feature maps ---
fig, axes = plt.subplots(1, 5, figsize=(18, 3))

# Panel 0 — original input image
# Permute from [C, H, W] → [H, W, C] so matplotlib can display it as RGB
# clamp(0, 1) prevents any colour values from going out of range
original_display = input_tensor[0].permute(1, 2, 0).clamp(0, 1)
axes[0].imshow(original_display)
axes[0].set_title("Original Input", fontweight='bold')
axes[0].axis('off')

# Panels 1–4 — the first 4 of the 16 output feature maps
# Each map shows which parts of the image activated a specific learned filter.
# Bright areas = strong response; dark areas = weak response.
for i in range(4):
    # .detach() removes the tensor from the autograd graph before numpy conversion
    axes[i + 1].imshow(feature_maps[0, i].detach(), cmap='viridis')
    axes[i + 1].set_title(f'Filter {i + 1} response')
    axes[i + 1].axis('off')

plt.suptitle(
    f'Feature maps after Conv2d → ReLU → MaxPool\n{source_label}',
    fontsize=12
)
plt.tight_layout()
plt.show()
```

## Additional Resources

- [PyTorch Conv2d Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [Understanding Convolutions Visualized](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)
- [Deep Residual Learning for Image Recognition (ResNet Paper)](https://arxiv.org/abs/1512.03385)
- [PyTorch `nn.Module` Beginner Guide](https://pytorch.org/tutorials/beginner/nn_tutorial.html)
