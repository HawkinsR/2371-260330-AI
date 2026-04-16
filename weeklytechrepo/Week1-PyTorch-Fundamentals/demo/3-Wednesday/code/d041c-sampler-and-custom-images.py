import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms
import os

"""
D041c: Local Image Classification with Random Sampling (Chihuahua vs Muffin)

Key Topics:
- Using torchvision.datasets.ImageFolder for local directory structures.
- Using RandomSampler to train on varied subsets of data each epoch.
- Building a simple Convolutional Neural Network (CNN) for real-world images.
"""

# 1. Configuration & Path Setup
# The demo assumes a 'doggo' directory is present with 'train' and 'test' subfolders.
DATA_ROOT = "doggo"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

# 2. Define Robust Transforms for Real-World Images
# Real-world images come in many sizes. We must resize and crop them 
# to a uniform shape (e.g., 224x224) for the neural network.
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet defaults
])

# 3. Load Dataset using ImageFolder
# ImageFolder automatically uses subfolder names (e.g., 'chihuahua', 'muffin') as labels.
if not os.path.exists(TRAIN_DIR):
    print(f"ERROR: Dataset directory '{TRAIN_DIR}' not found.")
    print("Please ensure the 'doggo' folder is extracted in the script's directory.")
    # Exit or provide dummy data for demo purposes? 
    # For now, we'll stop to avoid confusing the user with crashes.
    import sys; sys.exit(1)

train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=data_transforms)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=data_transforms)

print(f"Classes found: {train_dataset.classes}") # ['chihuahua', 'muffin']
print(f"Total training images available: {len(train_dataset)}")

# 4. Implement Random Sampling
# Instead of seeing all images every epoch, we want to see only 200 random ones.
# This keeps the training pass fast while ensuring variety.
train_sampler = RandomSampler(
    train_dataset, 
    num_samples=200, # Only draw 200 samples per loop
    replacement=True  # Required when num_samples is used
)

# Pass the sampler to the DataLoader instead of 'shuffle=True'
train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 5. Define a Simple CNN
# CNNs are better for images because they 'scan' for patterns like ears, fur, or muffin tops!
class DoggoModel(nn.Module):
    def __init__(self):
        super(DoggoModel, self).__init__()
        self.features = nn.Sequential(
            # Simple Conv layer to detect edges/textures
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # Reduces 224x224 -> 112x112
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Reduces 112x112 -> 56x56
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 2) # 2 classes: Chihuahua or Muffin
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = DoggoModel()
print("\n--- Model Summary ---")
print(model)

# 6. Training & Evaluation Loops
def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    model.train()
    print(f"\n--- Epoch {epoch+1} (Sampling {len(dataloader.sampler)} images) ---")
    
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 2 == 0:
            print(f"  Batch {batch}: Loss = {loss.item():>7f}")

def evaluate(dataloader, model, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # We only evaluate the first batch for speed in this demo
            break
            
    print(f"  Evaluation: Accuracy = {100 * correct / 32:>0.1f}%")

# 7. Run Demo
# Configuration for the instructor:
NUM_EPOCHS = 3 
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("\n--- 3. Starting Training Loop ---")
for epoch in range(NUM_EPOCHS):
    train_loop(train_loader, model, criterion, optimizer, epoch)
    evaluate(test_loader, model, criterion)

print("\n--- Summary ---")
print("1. ImageFolder automatically mapped subdirectories to labels.")
print(f"2. RandomSampler ensured we only processed 200 random images per epoch across {NUM_EPOCHS} epochs.")
print("3. Each epoch saw a DIFFERENT random subset, illustrating efficient data usage.")
print("4. model.train() and model.eval() were used to manage model state correctly.")
