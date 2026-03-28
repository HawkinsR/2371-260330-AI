import torch
import torch.nn as nn
import torch.optim as optim

# Toy Dataset
# X: Time on site (minutes)
# y: Purchased (1) or Did Not Purchase (0)
X = torch.tensor([[0.5], [1.0], [2.5], [5.0], [8.0], [12.0]], dtype=torch.float32)
y = torch.tensor([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]], dtype=torch.float32)

class PurchaseClassifier(nn.Module):
    """
    Task 1: Build the Model Architecture
    """
    def __init__(self):
        # TODO: Initialize the parent class
        
        # TODO: Define a linear layer (1 input -> 1 output)
        pass
        
    def forward(self, x):
        # TODO: Route x through the linear layer and return it
        return None

def train_classifier():
    """
    Task 2: Train the Model with Scheduling
    """
    # 1. Instantiate the model
    model = None # TODO: Instantiate PurchaseClassifier
    
    # 2. Define Loss, Optimizer, and Scheduler
    # TODO: Use BCEWithLogitsLoss
    criterion = None
    
    # TODO: Use Adam optimizer with lr=0.5
    optimizer = None
    
    # TODO: Use StepLR, step_size=20, gamma=0.5
    scheduler = None
    
    epochs = 100
    
    print("--- Starting Training ---")
    for epoch in range(epochs):
        model.train()
        
        # --- THE 5 STEPS OF TRAINING ---
        # 1. Forward pass
        predictions = None
        
        # 2. Compute Loss
        loss = None
        
        # 3. Zero gradients
        
        
        # 4. Backward pass
        
        
        # 5. Optimizer step
        
        
        # --- SCHEDULER STEP ---
        # TODO: Step the scheduler
        
        
        if (epoch + 1) % 20 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | LR: {current_lr:.4f}")
            
    # Evaluation / Inference
    model.eval()
    with torch.no_grad():
        test_low = torch.tensor([[0.5]])
        test_high = torch.tensor([[10.0]])
        
        pred_low = model(test_low).item()
        pred_high = model(test_high).item()
        
        print("\n--- Final Testing ---")
        print(f"Raw Output (Logit) for 0.5 mins: {pred_low:.4f}")
        print(f"Raw Output (Logit) for 10.0 mins: {pred_high:.4f}")

if __name__ == "__main__":
    train_classifier()
