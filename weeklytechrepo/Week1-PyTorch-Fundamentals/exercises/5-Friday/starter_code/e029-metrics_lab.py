import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Simulated Imbalanced Dataset
X_train = torch.randn(100, 5)
y_train = torch.randint(0, 2, (100, 1)).float()
# Make it imbalanced: 90% zeros intentionally
y_train[0:90] = 0.0

X_val = torch.randn(20, 5)
y_val = torch.randint(0, 2, (20, 1)).float()
y_val[0:18] = 0.0

class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

def run_metrics_lab():
    model = BasicModel()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    print("--- Initializing TensorBoard ---")
    # TODO: Initialize the SummaryWriter at 'runs/metrics_lab'
    writer = None 
    
    epochs = 50
    for epoch in range(epochs):
        # --- TRAINING phase ---
        model.train()
        optimizer.zero_grad()
        train_preds = model(X_train)
        train_loss = criterion(train_preds, y_train)
        train_loss.backward()
        optimizer.step()
        
        # --- VALIDATION phase ---
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val)
            
            # Convert probabilities to hard binary predictions (0 or 1)
            binary_preds = (val_preds > 0.5).float()
            
            # TODO: Calculate TP, FP, and FN using basic tensor math
            # Hint for TP: count where binary_preds == 1 AND y_val == 1
            TP = 0.0
            FP = 0.0
            FN = 0.0
            
            # Avoid divide-by-zero errors safely
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            
            # TODO: Calculate F1 Score
            f1_score = 0.0 
        
        # --- TENSORBOARD LOGGING ---
        # TODO: Log train_loss as 'Loss/Train'
        
        # TODO: Log val_loss as 'Loss/Validation'
        
        # TODO: Log f1_score as 'Metrics/F1_Score'
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Val Loss: {val_loss:.4f} | F1: {f1_score:.4f}")

    print("--- Shutting down TensorBoard ---")
    # TODO: Flush and close the writer
    
    print("\nTraining complete! Run 'tensorboard --logdir=runs' to view.")

if __name__ == "__main__":
    run_metrics_lab()
