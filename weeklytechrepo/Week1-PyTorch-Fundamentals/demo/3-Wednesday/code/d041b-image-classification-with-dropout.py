import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

"""
D041b: Image Classification with Dropout (FashionMNIST)

This demo expands on the basic Dataset/DataLoader patterns.
Key Topics:
- Using real image datasets (FashionMNIST).
- Regularization with nn.Dropout.
- Why model.train() and model.eval() are mandatory.
- Connection to Logistic Regression (this is a multi-layer extension).
"""

# 1. Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Define Transforms
# Normalization is critical for deep learning. It shifts pixel values 
# from [0, 1] to roughly [-1, 1], which helps gradients flow more smoothly!
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,)) 
])

# 3. Load Real Data (FashionMNIST)
# If Logistic Regression is our "Hello World", FashionMNIST is the standard next step.
print("--- 1. Downloading/Loading FashionMNIST Dataset ---")
train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

# DataLoaders handle the batching and shuffling automatically.
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples.")

# 4. Define the Model
# Thinking back to Logistic Regression:
# - Logistic Regression = Linear -> Activation (like Sigmoid/Softmax)
# - This Model = Linear -> ReLU -> Dropout -> Linear -> (CrossEntropy internally applies Softmax)
class FashionModel(nn.Module):
    def __init__(self):
        super(FashionModel, self).__init__()
        self.flatten = nn.Flatten()
        self.model_layers = nn.Sequential(
            # Hidden layer with 128 neurons
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            
            # --- DROPOUT LAYER ---
            # Randomly 'mutes' 20% of neurons during each training step.
            # This prevents the model from getting 'lazy' and relying on just a few pixels.
            nn.Dropout(p=0.2), 
            
            # Final output layer (one unit per clothing category)
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.model_layers(x)

model = FashionModel().to(device)
print("\n--- 2. Model Architecture ---")
print(model)

# 5. Training and Evaluation Phases
# This is WHERE Dropout makes the biggest difference.

def train_one_epoch(model, dataloader, optimizer, criterion):
    # MANDATORY: .train() makes Dropout ACTIVE.
    model.train() 
    print("\n[Training] model.train() called. Dropout is now active (regularizing...)")
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = criterion(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            print(f"  Batch {batch:>3}: Loss = {loss.item():>7f}")
            if batch >= 300: break # Keep demo short

def evaluate(model, dataloader, criterion):
    # MANDATORY: .eval() turns Dropout OFF.
    # We want to test the model with ALL its learned 'knowledge' active.
    model.eval()
    print("\n[Evaluation] model.eval() called. Dropout is now disabled (full power!)")
    
    test_loss, correct = 0, 0
    with torch.no_grad(): # No need to track history for math we aren't differentiating
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            break # Just testing the first batch for speed
            
    print(f"  Evaluation Score: Accuracy = {100 * correct / 64:>0.1f}%")

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 6. Execution
print("\n--- 3. Running Demo Pass ---")
train_one_epoch(model, train_loader, optimizer, criterion)
evaluate(model, test_loader, criterion)

print("\n--- Key Takeaways ---")
print("- We used REAL images from FashionMNIST.")
print("- Dropout was added to the hidden layer to improve generalization.")
print("- We manually switched between .train() and .eval() modes.")
print("- This architecture is effectively 'Stacked' Logistic Regression with Dropout.")
