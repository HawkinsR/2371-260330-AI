"""
Demo: Advanced Training Mechanics
This script demonstrates how to set rigorous reproducibility seeds, use 
CUDA Automatic Mixed Precision (AMP) to speed up training, apply Gradient 
Clipping to stabilize learning, and implement a custom Early Stopping hook.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler # Core libraries for Mixed Precision Training
import numpy as np
import random

def set_seed(seed=42):
    """Locks all sources of randomness for deterministic runs."""
    # When debugging AI, we often want the exact same "random" initialization every time we run the script
    print(f"Setting random seed to {seed}...")
    random.seed(seed)                # Lock standard Python random
    np.random.seed(seed)             # Lock NumPy random
    torch.manual_seed(seed)          # Lock PyTorch CPU random
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # Lock PyTorch GPU random across all available GPUs
        torch.backends.cudnn.deterministic = True  # Force CuDNN to use deterministic algorithms
        torch.backends.cudnn.benchmark = False     # Disable auto-tuner which can introduce randomness

class EarlyStopping:
    """A custom callback that halts training when validation loss stops improving."""
    def __init__(self, patience=3, verbose=True):
        self.patience = patience # How many epochs to wait after last time validation loss improved
        self.verbose = verbose
        self.counter = 0         # Tracks how many epochs we've waited without improvement
        self.best_loss = None    # The lowest validation loss we've seen so far
        self.early_stop = False  # The flag that will tell the main loop to break

    def __call__(self, val_loss, model, epoch):
        # If this is the very first epoch, just save the score as the baseline
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch)
            
        # If the loss got WORSE (or stayed exactly the same)
        elif val_loss >= self.best_loss:
            self.counter += 1 # Increase our patience counter
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                
            # If we've hit our patience limit, flip the flag to stop training!
            if self.counter >= self.patience:
                self.early_stop = True
                
        # If the loss got BETTER, save the model and reset the counter!
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        if self.verbose:
            print(f"Validation loss decreased. Saving model checkpoint (Epoch {epoch})...")
        # In a real scenario, you'd un-comment the line below to save the weights to disk
        # torch.save(model.state_dict(), 'checkpoint.pt')

def demonstrate_advanced_mechanics():
    print("--- Advanced Training Mechanics ---")
    
    # 1. Reproducibility
    set_seed(42) # Ensure every student running this gets the exact same "random" numbers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    # 2. Setup Model, Data, and AMP Scaler
    model = nn.Sequential(
        nn.Linear(20, 50),
        nn.ReLU(),
        nn.Linear(50, 2)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # GradScaler is essential for mixed precision backprop
    # Mixed precision uses Float16 (half-precision) instead of Float32, making the model faster
    # However, Float16 gradients can be so tiny they become exactly 0.0 ("underflow").
    # The GradScaler multiplies the loss to make gradients bigger before they become 0.0!
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    # Early Stopping Callback
    early_stopping = EarlyStopping(patience=3, verbose=True)
    
    # 3. Simulated Training Loop
    epochs = 15
    print("\nStarting Training Loop...")
    for epoch in range(1, epochs + 1):
        # --- Simulate a Batch ---
        inputs = torch.randn(32, 20).to(device)
        targets = torch.randint(0, 2, (32,)).to(device)
        
        # Zero gradients explicitly
        optimizer.zero_grad()
        
        # FORWARD PASS: Cast operations to float16 dynamically
        # 'autocast' automatically converts safe operations to fast Float16 while keeping risky ops in Float32
        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        # BACKWARD PASS: Scale loss to prevent gradients from underflowing
        scaler.scale(loss).backward()
        
        # GRADIENT CLIPPING: Prevents the "Exploding Gradient" problem where updates are too large
        # First, we unscale the gradients back down to normal size
        scaler.unscale_(optimizer)
        # Then we clip them so no single gradient vector exceeds a size of 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # OPTIMIZER STEP: Step the weights and update the scaler's internal multiplier
        scaler.step(optimizer)
        scaler.update()
        
        # --- End of Batch ---
        
        if epoch % 1 == 0:
            print(f"Epoch [{epoch}/{epochs}], Train Loss: {loss.item():.4f}")
            
        # --- Verification Phase (Simulated Validation) ---
        # For demo purposes, we randomly simulate a validation loss that improves, but then stops improving
        val_loss = 0.5 - (epoch * 0.02) + random.uniform(0.0, 0.1)
        if epoch > 5:
            val_loss += epoch * 0.05 # Force the loss to go back up to trigger early stopping
            
        # Trigger the early stopping callback evaluate the current val_loss
        early_stopping(val_loss, model, epoch)
        
        # Check if the callback set the flag to Halt
        if early_stopping.early_stop:
            print("\n!!! Early stopping triggered. Halting training loop to prevent overfitting !!!")
            break
            
    print("-" * 50)

if __name__ == "__main__":
    demonstrate_advanced_mechanics()
