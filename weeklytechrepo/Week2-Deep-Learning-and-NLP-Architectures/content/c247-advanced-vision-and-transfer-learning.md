# Advanced Vision and Transfer Learning

## Learning Objectives

- Define advanced vision tasks beyond classification: Object Detection and Semantic Segmentation.
- Evaluate vision models using spatial overlap metrics: Intersection over Union (IoU) and the Dice Coefficient.
- Understand why Transfer Learning is the standard approach in industry instead of training from scratch.
- Implement Transfer Learning by freezing early layers and fine-tuning a custom classification head.
- Save and load model weights using PyTorch's `state_dict` structure.

## Why This Matters

Training a deep CNN from scratch requires massive datasets and weeks of GPU compute. ResNet-50, for example, was trained on 1.2 million images (ImageNet). In practice, engineers almost never start from scratch. Transfer Learning lets you download a model that already knows how to detect edges, textures, and shapes, then adapt just its final layers to your specific task. This approach works reliably with datasets of only a few hundred images and trains in minutes rather than weeks.

## The Concept

### Object Detection and Semantic Segmentation

Standard image classification outputs one label for the whole image. More advanced tasks require finer-grained predictions:

- **Object Detection:** Locates individual objects within a scene by predicting bounding boxes around them. A street camera might detect five pedestrians, two cars, and a traffic sign in a single frame.
- **Semantic Segmentation:** Classifies every individual pixel in an image. Instead of drawing a box around a road, it labels each pixel as "road," "sidewalk," "sky," and so on. This precision is required for autonomous driving.

### IoU and Dice Coefficient

Standard accuracy fails for segmentation. If a scene is 95% background and a model labels every pixel as "background," it achieves 95% accuracy while being completely wrong on the objects that matter. Instead, we use metrics that measure how well the predicted mask overlaps with the true mask.

> **Key Term - Intersection over Union (IoU):** A spatial overlap metric calculated as: (area of overlap between prediction and ground truth) / (combined area of both). A perfect prediction has an IoU of 1.0; no overlap at all gives 0.0. Used as the standard metric for object detection and segmentation tasks.

> **Key Term - Dice Coefficient:** An alternative spatial overlap metric calculated as `(2 × overlap) / (predicted area + actual area)`. It gives more weight to the overlapping region, making it more sensitive to small targets. It is the preferred metric in medical imaging, where the region of interest (e.g., a tumor) is small relative to the full image.

### Transfer Learning via `state_dict`

A PyTorch `state_dict` is a Python dictionary mapping each layer name to its learned weights and biases. When you download a pre-trained model from PyTorch or Hugging Face, you are downloading a `state_dict` that has already been trained on millions of examples.

The Transfer Learning workflow:
1. **Load** a pre-trained model with its weights.
2. **Freeze** the early convolutional layers by setting `requires_grad = False`. These layers already know how to detect edges and textures—we do not want our small dataset to overwrite them.
3. **Replace** the final classification head with a new layer matching our number of output classes.
4. **Fine-tune** — only the new head (and optionally the last few unfrozen layers) will be updated during training.

**Checkpointing** is the practice of saving the `state_dict` after each epoch. This allows long training runs to be resumed after an interruption and enables rolling back to the best-performing checkpoint.

> **Key Term - Transfer Learning:** Initializing a model with weights pre-trained on a large general dataset (like ImageNet) rather than training from random weights. The pre-trained model already has useful visual representations in its early layers, which transfer well to new tasks with smaller datasets.

> **Key Term - Fine-Tuning:** The step within a Transfer Learning workflow where some or all of the pre-trained layers are unfrozen and trained at a very low learning rate on the new dataset. This adapts the model's knowledge to the specific task without overwriting what it learned during pre-training.

## Code Example

```python
import torch
import torch.nn as nn
import torchvision.models as models

# 1. Load a pre-trained ResNet18 — downloads ImageNet weights
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 2. Freeze all layers — we don't want to overwrite the pre-trained features
for param in model.parameters():
    param.requires_grad = False

# 3. Replace the final layer with one that matches our number of output classes
# ResNet18's final layer is named 'fc' and expects 512 input features
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # New head for 3 custom classes
# Note: nn.Linear sets requires_grad=True by default, so only this layer trains

# 4. Save and load the state_dict (checkpointing)
torch.save(model.state_dict(), 'checkpoint.pth')

# To resume training or run inference from a checkpoint:
new_model = models.resnet18()
new_model.fc = nn.Linear(num_ftrs, 3)
new_model.load_state_dict(torch.load('checkpoint.pth'))
new_model.eval()  # Always call eval() before inference to disable dropout
```

## Additional Resources

- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
