# Advanced Vision and Transfer Learning

## Learning Objectives

- Define advanced vision tasks beyond classification: Object Detection and Semantic Segmentation.
- Evaluate vision models using spatial overlap metrics: Intersection over Union (IoU) and the Dice Coefficient.
- Understand why Transfer Learning is the standard approach in industry instead of training from scratch.
- Implement Transfer Learning by freezing early layers and fine-tuning a custom classification head.
- Save and load model weights using PyTorch's `state_dict` structure.
- Explain how `nn.Embedding` layers map word tokens to dense numeric vectors.
- Handle variable-length sequences in mini-batches using `pack_padded_sequence` and `pad_packed_sequence`.

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

> **Key Term - Dice Coefficient:** An alternative spatial overlap metric calculated as `(2 × overlap) / (predicted area + actual area)`. It gives more weight to the overlapping region, making it more sensitive to small targets. It is the preferred metric in fields like medical imaging, where the region of interest (e.g., a tumor) is small relative to the full image.

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

```python
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter

DATA_ROOT = "data/doggos"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LOG_DIR = "runs/doggo_logs"
MODEL_PATH = "doggo.pth"

if not os.path.exists(TRAIN_DIR):
    print(f"ERROR: Dataset directory '{TRAIN_DIR}' not found.")
    print("Please ensure the 'doggo' folder is extracted in the script's directory.")
    sys.exit(1)

data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=data_transforms)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=data_transforms)

print(f"Classes found: {train_dataset.classes}")
print(f"Total training images available: {len(train_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

class PreTrainedModel(nn.Module):
    def __init__(self):
        super(PreTrainedModel, self).__init__()

        # Load ResNet18 with weights pre-trained on ImageNet
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Replace the final layer with one that matches our number of output classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.model(x)

class DoggoModel(nn.Module):
    def __init__(self):
        super(DoggoModel, self).__init__()
        self.flatten = nn.Flatten()

        self.features = nn.Sequential(
            # nn.Conv2d is a 2D convolution layer, slides a kernel over the input image
            # nn.MaxPool2d is a 2D max pooling layer, reduces the spatial dimensions of the input image
            # nn.ReLU is a rectified linear unit (ReLU) activation function, introduces non-linearity

            # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
            nn.Conv2d(3, 16, kernel_size=3, stride=1), # Convolute 3x3 kernel, stepping by 1
            nn.ReLU(), 
            nn.MaxPool2d(2), # 256x256 -> 128x128

            nn.Conv2d(16, 32, kernel_size=3, stride=1), 
            nn.ReLU(),
            nn.MaxPool2d(2), # 128x128 -> 64x64
        )

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 62 * 62, 128), # 32 filters, 62x62 pixels, 128 neurons
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2) # 128 neurons, 2 outputs
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classify(x)
        return x

def train_loop(dataloader, model, loss_fn, optimizer, epoch, best_loss, writer, device):
    print()

    print(f"\n--- Training Epoch {epoch+1} ---")
    
    model.train()
    start_time = time.time()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", loss.item(), batch)
        
        # print(f"Batch {batch}: Loss = {loss.item():>7f}")

        if loss < best_loss:
            best_loss = loss

            print("New best model found! Loss: ", loss.item(), " Saving...")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, MODEL_PATH)

        if batch % 100 == 0:
            print(f"Batch {batch}: Loss = {loss.item():>7f}")

    end_time = time.time()
    print(f"Epoch {epoch+1} completed: {batch+1} batches processed")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    return model, best_loss

def evaluate(dataloader, model, loss_fn, writer, device):
    print()
    print("--- Eval Model ---")

    test_loss, correct, total= 0, 0, 0

    model.eval()

    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total += len(y)
            test_loss += loss_fn(pred, y).item()
            correct += int((pred.argmax(1) == y).type(torch.float).sum().item())
            if batch == 9: break
    
    writer.add_scalar("Loss/test", test_loss / total)
    
    print("Total Samples: ", total)
    print("Correct Predictions: ", correct)
    print(f"Test Loss: {test_loss / total:.4f}")
    print(f"Evaluation: Accuracy = {int(100 * correct / total)}%" )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)

    print()
    print("--- Tensorboard Setup---")
    writer = SummaryWriter(LOG_DIR)

    print()
    print("--- Instantiate Model ---")
#   model = DoggoModel()
    model = PreTrainedModel().to(device)
    best_loss = float('inf')
    
    print("Adding graph to tensorboard...")
    dummy_data = torch.randn(1, 3, 256, 256).to(device)
    writer.add_graph(model, dummy_data)

    NUM_EPOCHS = 1
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.001
    )
    criterion = nn.CrossEntropyLoss()

    print("--- Load Best Model ---")
    if os.path.exists(MODEL_PATH):
        best_model = torch.load(MODEL_PATH, weights_only=True)
        model.load_state_dict(best_model['model_state_dict'])
        optimizer.load_state_dict(best_model['optimizer_state_dict'])
        best_loss = best_model['loss']
        print("Loaded best model from ", MODEL_PATH)

    for epoch in range(NUM_EPOCHS):
        model, best_loss = train_loop(train_loader, model, criterion, optimizer, epoch, best_loss, writer, device)
        evaluate(test_loader, model, criterion, writer, device)

if __name__ == "__main__":
    main()
```

## NLP Deepening: Embeddings and Variable-Length Sequences

Transfer Learning is not exclusive to Computer Vision. This section deepens the NLP foundations introduced on Tuesday, focusing on the mechanics that make real-world text batches work.

### `nn.Embedding` in Depth

On Tuesday, we saw that `nn.Embedding` maps word integer IDs to dense vectors. It's worth understanding *how* this works internally: an Embedding layer is simply a lookup table — a matrix of shape `[vocab_size, embedding_dim]`. When you pass in a token ID (e.g., `42`), it retrieves row 42 from that matrix. These rows are learned parameters, updated during backpropagation just like any other weight.

> **Key Term - Embedding Matrix:** The internal weight matrix of an `nn.Embedding` layer, shaped `[vocab_size, embedding_dim]`. Each row is a learned numeric vector for one word. Words that appear in similar training contexts end up with geometrically similar rows, giving the model a built-in understanding of semantic similarity.

### Handling Variable-Length Sequences

Real text in a mini-batch is never uniform. A batch of 32 sentences might have lengths ranging from 4 to 87 words. To process these efficiently on a GPU (which requires fixed-size tensors), shorter sequences are **padded** with a special `<PAD>` token to match the longest sequence in the batch. But this creates a problem: the LSTM will waste computation processing padding tokens, and the hidden state will be polluted by padding positions.

PyTorch provides two utilities to solve this:

- **`pack_padded_sequence`:** Takes a padded batch and a list of true sequence lengths. It collects only the real (non-padding) tokens and packs them into a compact `PackedSequence` object. The LSTM processes only real tokens at each time step, skipping padding completely.
- **`pad_packed_sequence`:** After the LSTM processes the `PackedSequence`, this function unpacks it back into a padded tensor of shape `[Batch, MaxSeqLen, HiddenDim]`, so you can access per-token outputs.

> **Key Term - Padding:** The practice of appending a special `<PAD>` token to shorter sequences in a batch to make them all the same length. Required for tensor batch operations, but must be handled carefully so that the model's loss function and recurrent layers ignore the padding positions.

> **Key Term - `PackedSequence`:** A PyTorch data structure (produced by `pack_padded_sequence`) that stores only the real tokens from a padded batch in a compact, sorted format. Passing a `PackedSequence` directly into `nn.LSTM` causes the LSTM to process only real content at each time step, eliminating wasted computation on padding.

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Vocabulary of 1000 words, 64-dim embeddings, 128-dim hidden LSTM
embedding = nn.Embedding(num_embeddings=1000, embedding_dim=64)
lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)

# Batch of 4 padded sequences, max length 10. Real lengths: [10, 7, 5, 3]
padded_batch = torch.randint(0, 1000, (4, 10))  # [Batch, MaxSeqLen]
lengths = torch.tensor([10, 7, 5, 3])           # True lengths, sorted descending

# Step 1: Embed — [Batch, MaxSeqLen] -> [Batch, MaxSeqLen, EmbeddingDim]
embedded = embedding(padded_batch)

# Step 2: Pack — removes padding and compacts all real tokens
packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=True)

# Step 3: Run LSTM on the PackedSequence (no padding waste)
packed_out, (h_n, c_n) = lstm(packed)

# Step 4: Unpack — restore padded shape [Batch, MaxSeqLen, HiddenDim]
output, output_lengths = pad_packed_sequence(packed_out, batch_first=True)

print("Output shape (padded):", output.shape)   # [4, 10, 128]
print("Final hidden state:", h_n.shape)          # [1, 4, 128]
```

## Additional Resources

- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [PyTorch Sequence-to-Sequence Tutorial (pack_padded_sequence)](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
