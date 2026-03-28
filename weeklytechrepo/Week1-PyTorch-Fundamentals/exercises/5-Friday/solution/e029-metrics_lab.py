import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Simulated Imbalanced Dataset
X_train = torch.randn(100, 5)
y_train = torch.randint(0, 2, (100, 1)).float()
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
    writer = SummaryWriter('runs/metrics_lab')
    
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_preds = model(X_train)
        train_loss = criterion(train_preds, y_train)
        train_loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val)
            
            binary_preds = (val_preds > 0.5).float()
            
            TP = torch.sum((binary_preds == 1.0) & (y_val == 1.0)).item()
            FP = torch.sum((binary_preds == 1.0) & (y_val == 0.0)).item()
            FN = torch.sum((binary_preds == 0.0) & (y_val == 1.0)).item()
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            
            f1_score = 0.0
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
        
        writer.add_scalar('Loss/Train', train_loss.item(), epoch)
        writer.add_scalar('Loss/Validation', val_loss.item(), epoch)
        writer.add_scalar('Metrics/F1_Score', f1_score, epoch)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Val Loss: {val_loss:.4f} | F1: {f1_score:.4f}")

    print("--- Shutting down TensorBoard ---")
    writer.flush()
    writer.close()
    print("\nTraining complete! Run 'tensorboard --logdir=runs' to view.")

if __name__ == "__main__":
    run_metrics_lab()
