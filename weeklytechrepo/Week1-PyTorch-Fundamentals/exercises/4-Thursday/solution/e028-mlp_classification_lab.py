import torch
import torch.nn as nn
import torch.optim as optim
import os

# Simulated IoT Sensor Dataset (4 categories)
X_train = torch.randn(200, 20) # 200 samples, 20 features
y_train = torch.randint(0, 4, (200,)) # 4 classes
X_val = torch.randn(50, 20)
y_val = torch.randint(0, 4, (50,))

class SensorMLP(nn.Module):
    def __init__(self):
        super(SensorMLP, self).__init__()
        self.fc1 = nn.Linear(20, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)
        
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x


def train_and_validate():
    model = SensorMLP()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 50
    best_val_loss = float('inf')
    
    print("--- Starting Hybrid Sensor Training ---")
    
    for epoch in range(epochs):
        
        # =======================
        #      TRAINING PHASE
        # =======================
        model.train()
        optimizer.zero_grad()
        
        train_preds = model(X_train)
        train_loss = criterion(train_preds, y_train)
        
        train_loss.backward()
        optimizer.step()
        
        # =======================
        #     VALIDATION PHASE
        # =======================
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
        
        # =======================
        #     CHECKPOINTING
        # =======================
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), "best_sensor_model.pth")
            print("  -> Saved new best model!")

    print("\n--- Training Complete ---")

if __name__ == "__main__":
    train_and_validate()
