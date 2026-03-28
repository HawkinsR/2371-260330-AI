import torch
import torch.nn as nn
import torch.optim as optim
import os

# 1. Simulate a Basic Dataset
# X_train: Generate 100 random samples, each with 10 features (e.g., 10 different sensor readings)
X_train = torch.randn(100, 10) 
# y_train: Generate 100 random binary labels (0 or 1, e.g., "Defective" or "Normal")
y_train = torch.randint(0, 2, (100,)) 

# Create a small unseen validation set to test if our model is memorizing or actually learning
X_val = torch.randn(20, 10)
y_val = torch.randint(0, 2, (20,))

# 2. Define the Multi-Layer Perceptron (MLP)
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        
        # Layer 1: The input layer mapping to a hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Activation Function: ReLU (Rectified Linear Unit) introduces non-linearity
        # Without non-linearity, a neural network is just a giant linear regression model
        self.relu = nn.ReLU() 
        
        # Regularization: Dropout randomly sets 50% of the neurons to 0 during training
        # This prevents the network from relying too heavily on any single neuron (overfitting)
        self.dropout = nn.Dropout(p=0.5) 
        
        # Layer 2 (Output): Maps the hidden layer to our final output classes (e.g., probability of 0 and 1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Defines the exact sequence in which data travels through the layers
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out) # Remember: Dropout only activates during training mode!
        out = self.fc2(out)
        return out

def run_training():
    # Instantiate the network (10 inputs -> 32 hidden neurons -> 2 output classes)
    model = SimpleMLP(input_dim=10, hidden_dim=32, output_dim=2)
    
    # CrossEntropyLoss is the gold standard for classification problems
    criterion = nn.CrossEntropyLoss() 
    
    # Adam optimizer intelligently adjusts the learning rate for each specific weight
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 10
    best_val_loss = float('inf') # Start with infinitely bad validation loss
    
    # Create checkpoints directory to save our model weights safely
    os.makedirs("checkpoints", exist_ok=True)
    
    print("--- Starting Complex Training Run ---")
    
    # 3. The Epoch Loop (An epoch is a full pass over the dataset)
    for epoch in range(epochs):
        
        # --- TRAINING PHASE ---
        # CRITICAL: model.train() turns ON layers like Dropout and BatchNorm 
        model.train() 
        
        # Clear old gradients from the previous step
        optimizer.zero_grad()
        
        # 1. Forward Pass
        train_outputs = model(X_train)
        
        # 2. Calculate Error
        train_loss = criterion(train_outputs, y_train)
        
        # 3. Backward Pass (Calculate Gradients)
        train_loss.backward()
        
        # 4. Update Weights
        optimizer.step()
        
        # --- VALIDATION PHASE ---
        # CRITICAL: model.eval() turns OFF Dropout. We want the full network's brainpower for testing.
        model.eval() 
        val_loss = 0.0
        
        # CRITICAL: torch.no_grad() disables Autograd tracking.
        # We don't need gradients for validation, so this saves a massive amount of memory and time.
        with torch.no_grad(): 
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
        
        # --- CHECKPOINTING ---
        # If this epoch performed better on the validation set than any previous epoch, save it!
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            print(f"  -> Validation loss decreased! Saving checkpoint...")
            
            # Save the 'state_dict' (which contains all the trained weights and biases)
            # We also save the epoch number and optimizer state so we could resume training later if we wanted
            checkpoint_path = "checkpoints/best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, checkpoint_path)

    print("\n--- Training Complete ---")
    print(f"Best Validation Loss recorded: {best_val_loss:.4f}")

if __name__ == "__main__":
    run_training()
