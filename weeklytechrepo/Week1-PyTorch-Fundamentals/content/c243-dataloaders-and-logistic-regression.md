# DataLoaders and Logistic Regression

## Learning Objectives

- Implement Logistic Regression for binary classification tasks.
- Build a custom Dataset class by implementing `__len__` and `__getitem__`.
- Use `DataLoader` to feed data to a model in shuffled mini-batches.
- Apply TorchVision Transforms including normalization as a preprocessing step.
- Identify class imbalance and understand strategies for handling it.

## Why This Matters

Real-world datasets are too large to fit in memory all at once. PyTorch's `Dataset` and `DataLoader` system solves this by reading data from disk in parallel on the CPU and feeding it to the GPU in manageable batches. This pattern—custom dataset, DataLoader, transforms—is used in virtually every PyTorch training pipeline.

## The Concept

### Custom Datasets

In PyTorch, a `Dataset` is a class that knows how to access individual samples from your data. To create one, you inherit from `torch.utils.data.Dataset` and implement two methods:

- `__len__`: Returns the total number of samples.
- `__getitem__(idx)`: Returns a single sample (and its label) given an index.

This design cleanly separates data loading from model logic. Your model doesn't need to know whether data is coming from local files, a cloud storage bucket, or a database.

> **Key Term - Dataset Interface:** A PyTorch contract that standardizes how data is accessed. Any class that correctly implements `__len__` and `__getitem__` can be used interchangeably with PyTorch's `DataLoader`, regardless of where the underlying data lives.

### DataLoaders and Transforms

The `DataLoader` wraps a `Dataset` and handles the mechanics of batching and shuffling. It can load multiple samples in parallel using worker processes, which prevents the CPU from becoming a bottleneck.

**Transforms** (from `torchvision.transforms`) are preprocessing functions applied to each sample as it is loaded. They are chained together using `transforms.Compose`. **Normalization** is the most common transform: it rescales pixel values to have a mean near 0 and a standard deviation near 1, which stabilizes gradient updates during training.

> **Key Term - DataLoader:** A PyTorch utility that wraps a Dataset and delivers data to your training loop in batches. It handles shuffling, parallel loading via worker processes, and assembling individual samples into batch tensors.

> **Key Term - Transforms:** Functions applied to each data sample during loading to preprocess or augment it. Common transforms include resizing images, converting them to tensors, and normalizing pixel values.

### Logistic Regression and Class Imbalance

Logistic Regression extends a basic linear model by passing the raw output through a **Sigmoid** (for binary classification) or **Softmax** (for multi-class) activation function. These functions squash unbounded numbers into a probability range of `[0, 1]`, making the output interpretable as a confidence score.

**Class imbalance** occurs when one class has far more examples than another. For example, a fraud detection dataset might be 99% legitimate transactions and 1% fraudulent. A model that always predicts "legitimate" would achieve 99% accuracy while being completely useless. Common solutions include oversampling the minority class or using a weighted loss function that penalizes errors on the minority class more heavily.

> **Key Term - Mini-Batch:** A small subset of the training data processed together in one forward/backward pass. Processing one sample at a time is too slow; processing the entire dataset at once requires too much memory. A mini-batch (typically 32–256 samples) strikes the balance between efficiency and stability.

> **Key Term - Sigmoid Function:** An activation function that maps any real number to a value between 0 and 1. It is used in binary classification to interpret a model's raw output as a probability. An output of `0.85` means the model is 85% confident the input belongs to the positive class. The shape of the function is a smooth "S" curve.

> **Key Term - Softmax Function:** An activation function for multi-class classification. It takes a list of raw scores (one per class) and converts them into a probability distribution that sums to exactly 1.0. The class with the highest probability is the predicted class.

> **Key Term - Class Imbalance:** A condition where the training dataset has significantly more examples of one class than another. This causes models to be biased toward the majority class and perform poorly on the minority class, even when overall accuracy appears high.

## Code Example

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Custom Dataset: defines how to access individual samples
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # Total number of samples
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load one sample by index (using random tensor here to simulate an image)
        image_tensor = torch.rand(3, 224, 224)  # Shape: [C, H, W]
        label = self.labels[idx]

        # Apply transforms if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, label

# Define a normalization transform using ImageNet mean and std values
transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the dataset and wrap it in a DataLoader
dataset = CustomImageDataset(
    image_paths=["a.jpg", "b.jpg", "c.jpg"],
    labels=[0, 1, 0],
    transform=transform
)

# shuffle=True randomizes the order each epoch; num_workers=2 loads data in parallel
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

# Training loop sketch
# for batch_images, batch_labels in dataloader:
#     predictions = model(batch_images)
#     loss = criterion(predictions, batch_labels)
#     ...
```

## Additional Resources

- [Datasets and DataLoaders Overview](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Mastering TorchVision Transforms](https://pytorch.org/vision/stable/transforms.html)
