import torch
import torch.nn as nn
import torch.optim as optim

# Toy Dataset
X = torch.tensor([[0.5], [1.0], [2.5], [5.0], [8.0], [12.0]], dtype=torch.float32)
y = torch.tensor([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]], dtype=torch.float32)

class PurchaseClassifier(nn.Module):
    def __init__(self):
        super(PurchaseClassifier, self).__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.linear(x)

def train_classifier():
    model = PurchaseClassifier()
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    epochs = 100
    
    print("--- Starting Training ---")
    for epoch in range(epochs):
        model.train()
        
        # 1. Forward pass
        predictions = model(X)
        
        # 2. Compute Loss
        loss = criterion(predictions, y)
        
        # 3. Zero gradients
        optimizer.zero_grad()
        
        # 4. Backward pass
        loss.backward()
        
        # 5. Optimizer step
        optimizer.step()
        
        # Scheduler Step
        scheduler.step()
        
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
        print(f"Raw Output (Logit) for 0.5 mins: {pred_low:.4f} (Negative/Low probability)")
        print(f"Raw Output (Logit) for 10.0 mins: {pred_high:.4f} (Positive/High probability)")

if __name__ == "__main__":
    train_classifier()
