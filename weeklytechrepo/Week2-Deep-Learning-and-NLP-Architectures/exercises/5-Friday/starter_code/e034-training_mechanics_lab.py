import torch
import torch.nn as nn
import random
import numpy as np
import copy

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None  # Will store a deep copy of the best model state_dict

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            # Save the initial weights as the first 'best'
            self.best_weights = copy.deepcopy(model.state_dict())
        elif val_loss < self.best_loss:
            # Validation loss improved — save new best weights and reset counter
            self.best_loss = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def robust_training_loop():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    ).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # 1. TODO: Initialize the EarlyStopping callback with patience=3
    early_stopper = None
    
    epochs = 15
    print("\nStarting Training Strategy...")
    for epoch in range(1, epochs + 1):
        
        # --- Simulate a Batch ---
        inputs = torch.randn(16, 10).to(device)
        targets = torch.randint(0, 2, (16,)).to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # 2. TODO: Implement Gradient Clipping
        # Use torch.nn.utils.clip_grad_norm_ on model.parameters() with max_norm=1.0
        
        
        optimizer.step()
        
        print(f"Epoch [{epoch}/{epochs}], Train Loss: {loss.item():.4f}")
            
        # --- Simulated Validation ---
        # The loss decreases for 5 epochs, then starts going up to simulate overfitting
        val_loss = 1.0 - (epoch * 0.1)
        if epoch > 5:
            val_loss += (epoch - 5) * 0.15 
            
        print(f"  -> Validation Loss: {val_loss:.4f}")
        
        # 3. TODO: Pass the val_loss AND the model to the early_stopper
        # (The updated EarlyStopping class needs the model to save its weights)
        
        
        # 4. TODO: Check if early_stopper.early_stop is True. If so break the loop.
        # Print "Early stopping triggered! Training halted to prevent overfitting"
        
        
    print("-" * 50)
    
    # 5. TODO: Restore the best model weights after the loop completes.
    # Use: model.load_state_dict(early_stopper.best_weights)
    # This ensures the model is in its best-performing state, not the final (overfit) state.
    
    
    print("Training complete. Model restored to best checkpoint.")

if __name__ == "__main__":
    robust_training_loop()
